import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
import models.common_model as MC


class RAW2RGBModel(BaseModel):
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
            "--n_resgroups", type=int, default=16, help="number of resgroup"
        )
        parser.add_argument("--nm", default=None)

        if "scale" not in args:
            parser.add_argument(
                "--scale", type=int, default=4, help="super resolution scale"
            )
        return parser

    def __init__(self, args):
        BaseModel.__init__(self, args)
        self.loss_names = ["content"]
        self.visual_names = ["lr", "sr", "hr"]

        self.fix_path = nn.Sequential(
            nn.Conv2d(
                4, self.args.n_feats, kernel_size=3, stride=1, padding=2, bias=True
            ),
            nn.PReLU(),
            nn.Conv2d(
                self.args.n_feats,
                self.args.n_feats,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.PReLU(),
            nn.Conv2d(
                self.args.n_feats,
                self.args.n_feats,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.PReLU(),
            nn.Conv2d(
                self.args.n_feats,
                self.args.n_feats,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.PReLU(),
            nn.Conv2d(
                self.args.n_feats,
                self.args.n_feats,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.PReLU(),
            nn.Conv2d(
                self.args.n_feats,
                self.args.n_feats * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                4, self.args.n_feats * 2, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.PReLU(),
        )

        self.downer = nn.Sequential(
            nn.Conv2d(
                self.args.n_feats * 2,
                self.args.n_feats * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.PReLU(),
            nn.Conv2d(
                self.args.n_feats * 2,
                self.args.n_feats * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
        )
        local_path = [
            MC.RBGroup(
                n_feats=self.args.n_feats * 4,
                nm=self.args.nm,
                n_blocks=self.args.n_blocks,
            )
            for _ in range(self.args.n_resgroups)
        ]
        local_path.append(
            nn.Conv2d(
                self.args.n_feats * 4,
                self.args.n_feats * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        )
        self.local_path = nn.Sequential(*local_path)
        self.uper = nn.Sequential(
            nn.ConvTranspose2d(
                self.args.n_feats * 4,
                self.args.n_feats * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.PReLU(),
            nn.ConvTranspose2d(
                self.args.n_feats * 2,
                self.args.n_feats * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
        )

        self.global_path = nn.Sequential(
            MC.MSRB(self.args.n_feats * 2),
            MC.MSRB(self.args.n_feats * 2),
            MC.MSRB(self.args.n_feats * 2),
            MC.MSRB(self.args.n_feats * 2),
            MC.MSRB(self.args.n_feats * 2),
            MC.MSRB(self.args.n_feats * 2),
            MC.MSRB(self.args.n_feats * 2),
            MC.MSRB(self.args.n_feats * 2),
        )
        self.global_down = nn.Conv2d(
            self.args.n_feats * 8 * 2,
            self.args.n_feats * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.linear = nn.Sequential(
            nn.Conv2d(
                self.args.n_feats * 4,
                self.args.n_feats * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.PReLU(),
        )

        self.tail = nn.Conv2d(
            self.args.n_feats * 2, 3, kernel_size=3, stride=1, padding=1, bias=True
        )

    def set_input(self, input):
        self.lr = input["lr"]
        self.hr = input["hr"]
        if self.args.gpu is not None:
            self.lr = self.lr.cuda(self.args.gpu, non_blocking=True)
            self.hr = self.hr.cuda(self.args.gpu, non_blocking=True)

    def forward(self, x):
        fix_s = F.interpolate(x, size=192, mode="bilinear")
        fix_s = self.fix_path(fix_s)

        x = self.head(x)

        x_down = self.downer(x)
        local_fea = self.local_path(x_down)
        local_fea += x_down
        local_fea = self.uper(local_fea)

        out = x
        msrb_out = []
        for i in range(8):
            out = self.global_path[i](out)
            msrb_out.append(out)
        global_fea = torch.cat(msrb_out, 1)
        global_fea = self.global_down(global_fea)

        fused_fea = torch.cat([global_fea, local_fea], 1)

        fused_fea = self.linear(fused_fea)
        fused_fea += fix_s

        x = self.tail(fused_fea)
        return F.tanh(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            self.args.lr,
            betas=(0.9, 0.999),
        )

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.start_epoch, gamma=0.1
        )

    def optimize_parameters(self):
        self.sr = self.forward(self.lr)
        self.mse_loss = F.mse_loss(self.sr, self.hr)
        self.optimizer.zero_grad()
        self.mse_loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return self.mse_loss, self.sr.size(0)
