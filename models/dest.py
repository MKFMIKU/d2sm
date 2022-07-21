# Deep Semantic Transfer: From Single-Semantic Image to Natural Image Restoration
# Author: Kangfu Mei
# Date: 03/30/2021
import torch
import torch.nn as nn
import torchvision.models as TM


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1,
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class DeST(nn.Module):
    def __init__(self, conv_index, height, width, K=224, Q=256, mean_shift=True):
        super(DeST, self).__init__()

        self.mean_shift = MeanShift()

        self.K = K
        self.Q = Q

        vgg_features = TM.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index.find("22") >= 0:
            self.feature_extracter = nn.Sequential(*modules[:8])
            self.feature_dim = 128 * height * width // 2 // 2
        elif conv_index.find("34") >= 0:
            self.feature_extracter = nn.Sequential(*modules[:17])
            self.feature_dim = 256 * height * width // 4 // 4
        elif conv_index.find("54") >= 0:
            self.feature_extracter = nn.Sequential(*modules[:35])
            self.feature_dim = 512 * height * width // 16 // 16

        # create the queue
        self.register_buffer("data_queue", torch.randn(K, self.feature_dim))
        self.register_buffer("label_queue", torch.randn(K, self.feature_dim))
        self.data_queue = nn.functional.normalize(self.data_queue, dim=1)
        self.label_queue = nn.functional.normalize(self.label_queue, dim=1)

        self.register_buffer("data_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_queue_ptr", torch.zeros(1, dtype=torch.long))

    def _encode(self, x):
        x = self.mean_shift(x)
        x = self.feature_extracter(x)
        x = x.view(x.size(0), -1)
        return nn.functional.normalize(x, dim=1)

    def _divergence(self, feat_s, feat_t, eps=1e-6):
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
        divergence = torch.mean(
            feat_t_cond_prob
            * torch.log((feat_t_cond_prob + eps) / (feat_s_cond_prob + eps))
        )
        return divergence

    @torch.no_grad()
    def _dequeue_and_enqueue_data(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.data_queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.data_queue[ptr : ptr + batch_size, :] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.data_queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_label(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.label_queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.label_queue[ptr : ptr + batch_size, :] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.label_queue_ptr[0] = ptr

    def forward(self, data, label):
        data_fea = self._encode(data)
        self._dequeue_and_enqueue_data(data_fea)
        with torch.no_grad():
            label_fea = self._encode(label)
            self._dequeue_and_enqueue_label(label_fea)

        divergence = self._divergence(
            self.data_queue, self.label_queue.clone().detach()
        )

        return divergence
