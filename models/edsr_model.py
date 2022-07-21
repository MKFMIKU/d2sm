import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
import models.common_model as CM


class EDSRModel(BaseModel):
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
        parser.add_argument(
            "--res_scale",
            type=float,
            default=1.0,
            help="scale the residual in residual block",
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

        self.mean = torch.Tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        self.std = torch.Tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(args.input_nc, args.n_feats, 3, padding=1)
        self.body = CM.make_layer(
            CM.ResidualBlockNoBN,
            args.n_blocks,
            mid_channels=args.n_feats,
            res_scale=args.res_scale,
        )
        self.conv_after_body = nn.Conv2d(args.n_feats, args.n_feats, 3, 1, 1)
        self.upsample = CM.UpsampleModule(args.scale, args.n_feats)
        self.conv_last = nn.Conv2d(args.n_feats, args.output_nc, 3, 1, 1, bias=True)

        self.configure_optimizers()

    def set_input(self, input):
        self.lr = input["lr"]
        self.hr = input["hr"]
        if self.args.gpu is not None:
            self.lr = self.lr.cuda(self.args.gpu, non_blocking=True)
            self.hr = self.hr.cuda(self.args.gpu, non_blocking=True)

    def forward(self, x):
        self.mean = self.mean.to(x)
        self.std = self.std.to(x)

        x = (x - self.mean) / self.std
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x * self.std + self.mean
        return x

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            self.args.lr,
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 200000 * 16 // self.args.batch_size, 0.5
        )

    def optimize_parameters(self, global_step):
        self.sr = self.forward(self.lr)
        self.content_loss = F.l1_loss(self.sr, self.hr)
        self.optimizer.zero_grad()
        self.content_loss.backward()
        self.optimizer.step()
        self.scheduler.step(global_step)
        return self.content_loss, self.sr.size(0)
