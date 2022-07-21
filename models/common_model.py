import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as TM


def default_init_weights(module, scale=1):
    """Initialize network weights.
    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            m.weight.data *= scale
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, val=1)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
    ::
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.
        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.
    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super(PixelShufflePack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2,
        )
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class UpsampleModule(nn.Sequential):
    """Upsample module used in EDSR.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        mid_channels (int): Channel number of intermediate features.
    """

    def __init__(self, scale, mid_channels):
        modules = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                modules.append(
                    PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
                )
        elif scale == 3:
            modules.append(
                PixelShufflePack(mid_channels, mid_channels, scale, upsample_kernel=3)
            )
        else:
            raise ValueError(
                f"scale {scale} is not supported. " "Supported scales: 2^n and 3."
            )

        super(UpsampleModule, self).__init__(*modules)


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
        sign=-1,
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class VGGLoss(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(VGGLoss, self).__init__()
        vgg_features = TM.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index.find("22") >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find("34") >= 0:
            self.vgg = nn.Sequential(*modules[:17])
        elif conv_index.find("54") >= 0:
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
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

        loss = F.l1_loss(vgg_sr, vgg_hr)

        return loss


class MSRB(nn.Module):
    def __init__(self, n_feats=64):
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = nn.Conv2d(
            n_feats, n_feats, kernel_size_1, stride=1, padding=kernel_size_1 // 2
        )
        self.conv_3_2 = nn.Conv2d(
            n_feats * 2,
            n_feats * 2,
            kernel_size_1,
            stride=1,
            padding=kernel_size_1 // 2,
        )
        self.conv_5_1 = nn.Conv2d(
            n_feats, n_feats, kernel_size_2, stride=1, padding=kernel_size_2 // 2
        )
        self.conv_5_2 = nn.Conv2d(
            n_feats * 2,
            n_feats * 2,
            kernel_size_2,
            stride=1,
            padding=kernel_size_2 // 2,
        )
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class RB(nn.Module):
    def __init__(self, n_feats, nm="in"):
        super(RB, self).__init__()
        module_body = []
        for i in range(2):
            module_body.append(
                nn.Conv2d(
                    n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True
                )
            )
            if nm == "in":
                module_body.append(nn.InstanceNorm2d(n_feats, affine=True))
            if nm == "bn":
                module_body.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                module_body.append(nn.PReLU())
        self.module_body = nn.Sequential(*module_body)

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return res


class RBGroup(nn.Module):
    def __init__(self, n_feats, n_blocks, nm="in"):
        super(RBGroup, self).__init__()
        module_body = [RB(n_feats, nm) for _ in range(n_blocks)]
        module_body.append(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.module_body = nn.Sequential(*module_body)

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return res


def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.
    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet
    Date:
        01/Jan/2019
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor, out_width, upscale_factor
    )

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PKTCosSim(nn.Module):
    """
    Learning Deep Representations with Probabilistic Knowledge Transfer
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf
    """

    def __init__(self):
        super(PKTCosSim, self).__init__()

    def forward(self, feat_s, feat_t, eps=1e-6):
        # Normalize each vector by its norm
        feat_s = feat_s.view(feat_s.size(0), -1)
        feat_t = feat_t.view(feat_t.size(0), -1)

        feat_s_norm = torch.sqrt(torch.sum(feat_s ** 2, dim=1, keepdim=True))
        feat_s = feat_s / (feat_s_norm + eps)
        feat_s[feat_s != feat_s] = 0

        feat_t_norm = torch.sqrt(torch.sum(feat_t ** 2, dim=1, keepdim=True))
        feat_t = feat_t / (feat_t_norm + eps)
        feat_t[feat_t != feat_t] = 0

        # Calculate the cosine similarity
        feat_s_cos_sim = torch.mm(feat_s, feat_s.transpose(0, 1))
        feat_t_cos_sim = torch.mm(feat_t, feat_t.transpose(0, 1))

        # Scale cosine similarity to [0,1]
        feat_s_cos_sim = (feat_s_cos_sim + 1.0) / 2.0
        feat_t_cos_sim = (feat_t_cos_sim + 1.0) / 2.0

        # Transform them into probabilities
        feat_s_cond_prob = feat_s_cos_sim / torch.sum(
            feat_s_cos_sim, dim=1, keepdim=True
        )
        feat_t_cond_prob = feat_t_cos_sim / torch.sum(
            feat_t_cos_sim, dim=1, keepdim=True
        )

        # Calculate the KL-divergence
        loss = torch.mean(
            feat_t_cond_prob
            * torch.log((feat_t_cond_prob + eps) / (feat_s_cond_prob + eps))
        )

        return loss


class DeepSemanticTransfer(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(DeepSemanticTransfer, self).__init__()
        vgg_features = TM.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index.find("22") >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find("34") >= 0:
            self.vgg = nn.Sequential(*modules[:17])
        elif conv_index.find("54") >= 0:
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False

        self.pkt_cossim = PKTCosSim()

    def unfold_th(self, im_th, kernel_size=224):
        kernel_size = 224
        N, C, _, _ = im_th.shape
        im_th_unfold = F.unfold(im_th, kernel_size=kernel_size, stride=kernel_size // 4)
        _, _, L = im_th_unfold.shape
        im_th_unfold = im_th_unfold.reshape(N, C, kernel_size, kernel_size, L)
        im_th_unfold = im_th_unfold.permute(0, 4, 1, 2, 3)
        im_th_unfold = im_th_unfold.reshape(N * L, C, kernel_size, kernel_size)
        return im_th_unfold

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.unfold_th(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = self.pkt_cossim(vgg_sr, vgg_hr)

        return loss
