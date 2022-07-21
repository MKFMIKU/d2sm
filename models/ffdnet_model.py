import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
import models.common_model as CM
import torchvision.models as TM
from models.DeST import DeST, MeanShift
import numpy as np
import math

def non_local_correlation(fea1, fea2):
    N,C,H,W = fea1.shape
    fea1 = fea1.reshape((N,C,H*W))
    fea2 = fea2.reshape((N,C,H*W))
    gram1 = torch.matmul(fea1, fea1.transpose(1,2)) / C*C
    gram2 = torch.matmul(fea2, fea2.transpose(1,2)) / C*C
    return F.l1_loss(gram1, gram2)


def nn_loss(reference, target, neighborhood_size=(5, 5)):
    v_pad = int(neighborhood_size[0] / 2)
    h_pad = int(neighborhood_size[1] / 2)
    reference = F.pad(reference, (v_pad, v_pad, h_pad, h_pad))
    reference_tensors = []
    for i_begin in range(0, neighborhood_size[0]):
        i_end = i_begin - neighborhood_size[0] + 1
        i_end = None if i_end == 0 else i_end
        for j_begin in range(0, neighborhood_size[1]):
            j_end = j_begin - neighborhood_size[0] + 1
            j_end = None if j_end == 0 else j_end
            sub_tensor = reference[:, :, i_begin:i_end, j_begin:j_end]
            reference_tensors.append(torch.unsqueeze(sub_tensor, -1))
    reference = torch.cat(reference_tensors, dim=-1)
    target = torch.unsqueeze(target, axis=-1)
    
    abs = torch.abs(reference - target)
    norms = torch.sum(abs, dim=1, keepdim=False)
    loss, _ = torch.min(norms, dim=-1, keepdim=False)
    loss = torch.mean(loss)
    return loss

class FFDNetModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        args, _ = parser.parse_known_args()
        parser.add_argument(
            "--input_nc",
            type=int,
            default=3,
            help="number of color channels for the input",
        )
        parser.add_argument(
            "--output_nc",
            type=int,
            default=3,
            help="number of color channels for the output",
        )
        parser.add_argument(
            "--n_feats", type=int, default=96, help="number of feature maps"
        )
        parser.add_argument(
            "--n_blocks", type=int, default=12, help="number of resblocks"
        )
        if "sigma" not in args:
            parser.add_argument(
                "--sigma", default=75, type=int, help="noise range of ffdnet"
            )
        # Deep-Semantic-Transfer related
        parser.add_argument(
            "--dest", action="store_true", help="if true uses Deep Semantic Transfer",
        )
        parser.add_argument(
            "--dest_q", type=int, default=1, help="queue size of memo-dest"
        )
        parser.add_argument(
            "--dest_k", type=int, default=1, help="patch size of inner-dest"
        )
        # Comparison related
        parser.add_argument(
            "--nlc", action="store_true", help="if true use Non-local correlation",
        )
        parser.add_argument(
            "--nnl", action="store_true", help="if true use Nearest neighbor loss",
        )
        parser.set_defaults(epochs=80)
        return parser

    def __init__(self, args):
        BaseModel.__init__(self, args)
        self.loss_names = ["content"]
        if self.args.dest:
            self.loss_names.append("dest")
            self.deep_semantic_transfer = DeST("54", args.patch_size, args.patch_size, Q=args.dest_q, K=args.dest_k, mean_shift=True)
        if self.args.nlc or self.args.nnl:
            self.loss_names.append("dest")
            self.mean_shift = MeanShift()
            self.vgg_features = nn.Sequential(*TM.vgg19(pretrained=True).features[:8])

        self.visual_names = ["noisy_hr", "hr", "denoised_hr"]

        self.sf = 2

        self.m_head = nn.Sequential(
            nn.Conv2d(
                self.args.input_nc * self.sf * self.sf + 1, self.args.n_feats, 3, 1, 1
            ),
            nn.ReLU(inplace=True),
        )

        self.m_body = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(self.args.n_feats, self.args.n_feats, 3, 1, 1),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.args.n_blocks - 2)
            ]
        )

        self.m_tail = nn.Conv2d(
            self.args.n_feats, self.args.output_nc * self.sf * self.sf, 3, 1, 1
        )

        self.configure_optimizers()

    def set_input(self, input):
        self.noisy_hr = input["noisy_hr"]
        self.hr = input["hr"]
        self.noise_level = input["noise_level"]
        if self.args.gpu is not None:
            self.noisy_hr = self.noisy_hr.cuda(self.args.gpu, non_blocking=True)
            self.hr = self.hr.cuda(self.args.gpu, non_blocking=True)
            self.noise_level = self.noise_level.cuda(self.args.gpu, non_blocking=True)

    def forward(self, x, sigma):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 2) * 2 - h)
        paddingRight = int(np.ceil(w / 2) * 2 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = CM.pixel_unshuffle(x, upscale_factor=self.sf)
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), 1)

        x = self.m_tail(self.m_body(self.m_head(x)))
        x = F.pixel_shuffle(x, upscale_factor=self.sf)

        x = x[..., :h, :w]
        return x

    def configure_optimizers(self):
        # From 1e-3 into 1e-4 into 1e-5 into 1e-6
        # multi_step_list = [40, 80, 120, 160, 200, 400]
        multi_step_list = [80, 160, 240, 320, 400, 800]
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            self.args.lr,
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, multi_step_list, 0.5,
        )

    def optimize_parameters(self, global_step):
        self.denoised_hr = self.forward(self.noisy_hr, self.noise_level)
        self.content_loss = F.l1_loss(self.denoised_hr, self.hr)
        if self.args.dest:
            self.dest_loss = self.deep_semantic_transfer(self.hr, self.denoised_hr)
            self.content_loss += self.dest_loss
        if self.args.nlc:
            with torch.no_grad():
                fea_hr = self.vgg_features(self.mean_shift(self.hr))
            fea_denoised_hr = self.vgg_features(self.mean_shift(self.denoised_hr))
            self.dest_loss = non_local_correlation(fea_hr, fea_denoised_hr) * 0.1
            self.content_loss += self.dest_loss
        if self.args.nnl:
            with torch.no_grad():
                fea_hr = self.vgg_features(self.mean_shift(self.hr))
            fea_denoised_hr = self.vgg_features(self.mean_shift(self.denoised_hr))
            self.dest_loss = nn_loss(fea_hr, fea_denoised_hr) * 0.1
            self.content_loss += self.dest_loss
        self.optimizer.zero_grad()
        self.content_loss.backward()
        self.optimizer.step()
        return self.content_loss, self.denoised_hr.size(0)

    def eval(self):
        with torch.no_grad():
            self.denoised_hr = self.forward(self.noisy_hr, self.noise_level)
        diff = self.denoised_hr - self.hr
        mse = diff.pow(2).mean()
        psnr = -10 * math.log10(mse)
        return psnr

    def test(self, x):
        _, C, H, W = x.shape
        np.random.seed(seed=0)
        noise = np.random.normal(0, self.args.sigma / 255.0, (H, W, C)).transpose(
            (2, 0, 1)
        )
        noise_x = x + torch.FloatTensor(noise).to(x.device).unsqueeze(0)
        noise_level = torch.FloatTensor([self.args.sigma / 255.0]).to(x.device)
        noise_level = noise_level.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        noise_level = noise_level.repeat(noise_x.size(0), 1, 1, 1)

        denoised_x = self.forward(noise_x, noise_level)
        return denoised_x
