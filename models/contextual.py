# Browser from https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from models import common_model


def contextual_oss(
    x: torch.Tensor, y: torch.Tensor, band_width: float = 0.5, loss_type: str = "cosine"
):
    """
    Computes contextual loss between x and y.
    The most of this code is copied from
        https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.
    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """

    assert x.size() == y.size(), "input tensor must have the same size."

    N, C, H, W = x.size()

    if loss_type == "cosine":
        dist_raw = compute_cosine_distance(x, y)
    elif loss_type == "l1":
        dist_raw = compute_l1_distance(x, y)
    elif loss_type == "l2":
        dist_raw = compute_l2_distance(x, y)

    dist_tilde = compute_relative_distance(dist_raw)
    cx = compute_cx(dist_tilde, band_width)
    cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # Eq(1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5))  # Eq(5)

    return cx_loss


# TODO: Operation check
def contextual_bilateral_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    weight_sp: float = 0.1,
    band_width: float = 1.0,
    loss_type: str = "cosine",
):
    """
    Computes Contextual Bilateral (CoBi) Loss between x and y,
        proposed in https://arxiv.org/pdf/1905.05169.pdf.
    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper).
    k_arg_max_NC : torch.Tensor
        indices to maximize similarity over channels.
    """

    assert x.size() == y.size(), "input tensor must have the same size."

    # spatial loss
    grid = compute_meshgrid(x.shape).to(x.device)
    dist_raw = compute_l2_distance(grid, grid)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_sp = compute_cx(dist_tilde, band_width)

    # feature loss
    if loss_type == "cosine":
        dist_raw = compute_cosine_distance(x, y)
    elif loss_type == "l1":
        dist_raw = compute_l1_distance(x, y)
    elif loss_type == "l2":
        dist_raw = compute_l2_distance(x, y)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_feat = compute_cx(dist_tilde, band_width)

    # combined loss
    cx_combine = (1.0 - weight_sp) * cx_feat + weight_sp * cx_sp

    k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)

    cx = k_max_NC.mean(dim=1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5))

    return cx_loss


def compute_cx(dist_tilde, band_width):
    w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
    cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
    return cx


def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde


def compute_cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # consine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

    # convert to distance
    dist = 1 - cosine_sim

    return dist


# TODO: Considering avoiding OOM.
def compute_l1_distance(x: torch.Tensor, y: torch.Tensor):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)

    dist = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
    dist = dist.sum(dim=1).abs()
    dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.0)

    return dist


# TODO: Considering avoiding OOM.
def compute_l2_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1, keepdim=True)
    y_s = torch.sum(y_vec ** 2, dim=1, keepdim=True)

    A = y_vec.transpose(1, 2) @ x_vec
    dist = y_s - 2 * A + x_s.transpose(1, 2)
    dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.0)

    return dist


def compute_meshgrid(shape):
    N, C, H, W = shape
    rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
    cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

    feature_grid = torch.meshgrid(rows, cols)
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

    return feature_grid


def contextual_loss(x, y, h=0.5):
    """Computes contextual loss between x and y.

    Args:
      x: features of shape (N, C, H, W).
      y: features of shape (N, C, H, W).
      h: hyper-parameters

    Returns:
      cx_loss = contextual loss between x and y (Eq (1) in the paper)
    """
    assert x.size() == y.size()
    (
        N,
        C,
        H,
        W,
    ) = (
        x.size()
    )  # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)

    x_centered = x - y_mu
    y_centered = y - y_mu
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # The equation at the bottom of page 6 in the paper
    # Vectorized computation of cosine similarity for each pair of x_i and y_j
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)
    d = 1 - cosine_sim  # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data

    d_min, _ = torch.min(d, dim=2, keepdim=True)  # (N, H*W, 1)
    # Eq (2)
    d_tilde = d / (d_min + 1e-6)

    # Eq(3)
    w = torch.exp((1 - d_tilde) / h)
    # Eq(4)
    cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)

    # Eq (1)
    cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
    cx_loss = torch.mean(-torch.log(cx + 1e-6))

    return cx_loss


class VGG(nn.Module):
    def __init__(self, rgb_range=1, args=None, type="l1"):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        modules = modules[:13]
        self.vgg = nn.Sequential(*modules)
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = common_model.MeanShift(rgb_range, vgg_mean, vgg_std)
        self.args = args
        self.type = type
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())
        if self.type == "l1":
            loss = F.l1_loss(vgg_sr, vgg_hr)
        elif self.type == "contextual":
            loss = contextual_loss(vgg_hr, vgg_sr, h=0.1)

        return loss
