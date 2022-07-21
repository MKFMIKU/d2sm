import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel


class SRCNNModel(BaseModel):
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
        if "scale" not in args:
            parser.add_argument(
                "--scale", type=int, default=4, help="super resolution scale"
            )
        return parser

    def __init__(self, args):
        BaseModel.__init__(self, args)
        self.loss_names = ["content"]
        self.visual_names = ["lr", "sr", "hr"]

        self.img_upsampler = nn.Upsample(
            scale_factor=self.args.scale, mode="bicubic", align_corners=False
        )

        self.conv1 = nn.Conv2d(
            self.args.input_nc, self.args.n_feats, kernel_size=3, padding=3 // 2
        )

        self.conv2 = nn.Conv2d(
            self.args.n_feats, self.args.n_feats // 2, kernel_size=3, padding=3 // 2
        )

        self.conv3 = nn.Conv2d(
            self.args.n_feats // 2, self.args.output_nc, kernel_size=3, padding=3 // 2
        )

        self.relu = nn.ReLU()

        self.configure_optimizers()

    def set_input(self, input):
        self.lr = input["lr"]
        self.hr = input["hr"]
        if self.args.gpu is not None:
            self.lr = self.lr.cuda(self.args.gpu, non_blocking=True)
            self.hr = self.hr.cuda(self.args.gpu, non_blocking=True)

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            self.args.lr,
            betas=(0.9, 0.999),
        )

    def optimize_parameters(self):
        self.sr = self.forward(self.lr)
        self.content_loss = F.l1_loss(self.sr, self.hr)
        self.optimizer.zero_grad()
        self.content_loss.backward()
        self.optimizer.step()
        return self.content_loss, self.sr.size(0)
