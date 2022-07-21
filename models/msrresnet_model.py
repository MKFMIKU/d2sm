import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
import models.common_model as MC


class MSRResNetModel(BaseModel):
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
            "--n_feats", type=int, default=64, help="number of feature maps"
        )
        parser.add_argument(
            "--n_blocks", type=int, default=16, help="number of resblocks"
        )
        if "scale" not in args:
            parser.add_argument(
                "--scale", type=int, default=4, help="super resolution scale"
            )
        return parser

    def __init__(self, args):
        BaseModel.__init__(self, args)
        self.loss_names = ["content"]
        self.visual_names = ["lr", "sr", "hr"]

        self.conv_first = nn.Conv2d(
            self.args.input_nc, self.args.n_feats, 3, 1, 1, bias=True
        )
        self.trunk_net = MC.make_layer(
            MC.ResidualBlockNoBN, self.args.n_blocks, mid_channels=self.args.n_feats
        )

        # upsampling
        if self.args.scale in [2, 3]:
            self.upsample1 = MC.PixelShufflePack(
                self.args.n_blocks,
                self.args.n_blocks,
                self.args.scale,
                upsample_kernel=3,
            )
        elif self.args.scale == 4:
            self.upsample1 = MC.PixelShufflePack(
                self.args.n_feats, self.args.n_feats, 2, upsample_kernel=3
            )
            self.upsample2 = MC.PixelShufflePack(
                self.args.n_feats, self.args.n_feats, 2, upsample_kernel=3
            )
        else:
            raise ValueError(
                f"Unsupported scale factor {self.upscale_factor}. "
                f"Currently supported ones are "
                f"{self._supported_upscale_factors}."
            )

        self.conv_hr = nn.Conv2d(
            self.args.n_feats, self.args.n_feats, 3, 1, 1, bias=True
        )
        self.conv_last = nn.Conv2d(
            self.args.n_feats, self.args.output_nc, 3, 1, 1, bias=True
        )

        self.img_upsampler = nn.Upsample(
            scale_factor=self.args.scale, mode="bilinear", align_corners=False
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.vgg_measurer = MC.VGGLoss(conv_index="22")

        self.configure_optimizers()

    def set_input(self, input):
        self.lr = input["lr"]
        self.hr = input["hr"]
        if self.args.gpu is not None:
            self.lr = self.lr.cuda(self.args.gpu, non_blocking=True)
            self.hr = self.hr.cuda(self.args.gpu, non_blocking=True)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.trunk_net(feat)

        if self.args.scale in [2, 3]:
            out = self.upsample1(out)
        elif self.args.scale == 4:
            out = self.upsample1(out)
            out = self.upsample2(out)

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        upsampled_img = self.img_upsampler(x)
        out += upsampled_img
        return out

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            self.args.lr,
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 250000 * 16 // self.args.batch_size, 1e-7
        )

    def optimize_parameters(self, global_step):
        self.sr = self.forward(self.lr)
        # self.content_loss = F.l1_loss(self.sr, self.hr)
        self.vgg_loss = self.vgg_measurer(self.sr, self.hr)
        self.optimizer.zero_grad()
        self.vgg_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return self.vgg_loss, self.sr.size(0)
