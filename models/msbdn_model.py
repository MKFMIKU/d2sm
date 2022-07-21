import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .dest import DeST


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            input_size, output_size, kernel_size, stride, padding, bias=bias
        )

        self.norm = norm
        if self.norm == "batch":
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == "instance":
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == "relu":
            self.act = nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = nn.PReLU()
        elif self.activation == "lrelu":
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != "no":
            return self.act(out)
        else:
            return out


class DeconvBlock(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            input_size, output_size, kernel_size, stride, padding, bias=bias
        )

        self.norm = norm
        if self.norm == "batch":
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == "instance":
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == "relu":
            self.act = nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = nn.PReLU()
        elif self.activation == "lrelu":
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class Encoder_MDCBlock1(nn.Module):
    def __init__(
        self,
        num_filter,
        num_ft,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=True,
        activation="prelu",
        norm=None,
        mode="iter1",
    ):
        super(Encoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.up_convs.append(
                DeconvBlock(
                    num_filter // (2 ** i),
                    num_filter // (2 ** (i + 1)),
                    kernel_size,
                    stride,
                    padding,
                    bias,
                    activation,
                    norm=None,
                )
            )
            self.down_convs.append(
                ConvBlock(
                    num_filter // (2 ** (i + 1)),
                    num_filter // (2 ** i),
                    kernel_size,
                    stride,
                    padding,
                    bias,
                    activation,
                    norm=None,
                )
            )

    def forward(self, ft_l, ft_h_list):
        if self.mode == "iter1" or self.mode == "conv":
            ft_l_list = []
            for i in range(len(ft_h_list)):
                ft_l_list.append(ft_l)
                ft_l = self.up_convs[self.num_ft - len(ft_h_list) + i](ft_l)

            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft_fusion = (
                    self.down_convs[self.num_ft - i - 1](ft_fusion - ft_h_list[i])
                    + ft_l_list[len(ft_h_list) - i - 1]
                )

        if self.mode == "iter2":
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == "iter3":
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(i + 1):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[len(ft_h_list) - i - 1]
                for j in range(i + 1):
                    # print(j)
                    ft = self.down_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == "iter4":
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_l
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion


class Decoder_MDCBlock1(nn.Module):
    def __init__(
        self,
        num_filter,
        num_ft,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=True,
        activation="prelu",
        norm=None,
        mode="iter1",
    ):
        super(Decoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.down_convs.append(
                ConvBlock(
                    num_filter * (2 ** i),
                    num_filter * (2 ** (i + 1)),
                    kernel_size,
                    stride,
                    padding,
                    bias,
                    activation,
                    norm=None,
                )
            )
            self.up_convs.append(
                DeconvBlock(
                    num_filter * (2 ** (i + 1)),
                    num_filter * (2 ** i),
                    kernel_size,
                    stride,
                    padding,
                    bias,
                    activation,
                    norm=None,
                )
            )

    def forward(self, ft_h, ft_l_list):
        if self.mode == "iter1" or self.mode == "conv":
            ft_h_list = []
            for i in range(len(ft_l_list)):
                ft_h_list.append(ft_h)
                ft_h = self.down_convs[self.num_ft - len(ft_l_list) + i](ft_h)

            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft_fusion = (
                    self.up_convs[self.num_ft - i - 1](ft_fusion - ft_l_list[i])
                    + ft_h_list[len(ft_l_list) - i - 1]
                )

        if self.mode == "iter2":
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == "iter3":
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(i + 1):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[len(ft_l_list) - i - 1]
                for j in range(i + 1):
                    # print(j)
                    ft = self.up_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == "iter4":
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_h
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(
            nChannels,
            growthRate,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, scale=1.0):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        self.scale = scale
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(
            nChannels_, nChannels, kernel_size=1, padding=0, bias=False
        )

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out) * self.scale
        out = out + x
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class MSBDN(nn.Module):
    def __init__(self, res_blocks=18):
        super(MSBDN, self).__init__()

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.dense0 = nn.Sequential(
            ResidualBlock(16), ResidualBlock(16), ResidualBlock(16)
        )

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.conv1 = RDB(16, 4, 16)
        self.fusion1 = Encoder_MDCBlock1(16, 2, mode="iter2")
        self.dense1 = nn.Sequential(
            ResidualBlock(32), ResidualBlock(32), ResidualBlock(32)
        )

        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv2 = RDB(32, 4, 32)
        self.fusion2 = Encoder_MDCBlock1(32, 3, mode="iter2")
        self.dense2 = nn.Sequential(
            ResidualBlock(64), ResidualBlock(64), ResidualBlock(64)
        )

        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.conv3 = RDB(64, 4, 64)
        self.fusion3 = Encoder_MDCBlock1(64, 4, mode="iter2")
        self.dense3 = nn.Sequential(
            ResidualBlock(128), ResidualBlock(128), ResidualBlock(128)
        )

        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.conv4 = RDB(128, 4, 128)
        self.fusion4 = Encoder_MDCBlock1(128, 5, mode="iter2")

        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module("res%d" % i, ResidualBlock(256))

        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(
            ResidualBlock(128), ResidualBlock(128), ResidualBlock(128)
        )
        self.conv_4 = RDB(64, 4, 64)
        self.fusion_4 = Decoder_MDCBlock1(64, 2, mode="iter2")

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3 = nn.Sequential(
            ResidualBlock(64), ResidualBlock(64), ResidualBlock(64)
        )
        self.conv_3 = RDB(32, 4, 32)
        self.fusion_3 = Decoder_MDCBlock1(32, 3, mode="iter2")

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2 = nn.Sequential(
            ResidualBlock(32), ResidualBlock(32), ResidualBlock(32)
        )
        self.conv_2 = RDB(16, 4, 16)
        self.fusion_2 = Decoder_MDCBlock1(16, 4, mode="iter2")

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1 = nn.Sequential(
            ResidualBlock(16), ResidualBlock(16), ResidualBlock(16)
        )
        self.conv_1 = RDB(8, 4, 8)
        self.fusion_1 = Decoder_MDCBlock1(8, 5, mode="iter2")

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)

    def forward(self, x):
        res1x = self.conv_input(x)
        res1x_1, res1x_2 = res1x.split(
            [(res1x.size()[1] // 2), (res1x.size()[1] // 2)], dim=1
        )
        feature_mem = [res1x_1]
        x = self.dense0(res1x) + res1x

        res2x = self.conv2x(x)
        res2x_1, res2x_2 = res2x.split(
            [(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1
        )
        res2x_1 = self.fusion1(res2x_1, feature_mem)
        res2x_2 = self.conv1(res2x_2)
        feature_mem.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)
        res2x = self.dense1(res2x) + res2x

        res4x = self.conv4x(res2x)
        res4x_1, res4x_2 = res4x.split(
            [(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1
        )
        res4x_1 = self.fusion2(res4x_1, feature_mem)
        res4x_2 = self.conv2(res4x_2)
        feature_mem.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)
        res4x = self.dense2(res4x) + res4x

        res8x = self.conv8x(res4x)
        res8x_1, res8x_2 = res8x.split(
            [(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1
        )
        res8x_1 = self.fusion3(res8x_1, feature_mem)
        res8x_2 = self.conv3(res8x_2)
        feature_mem.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)
        res8x = self.dense3(res8x) + res8x

        res16x = self.conv16x(res8x)
        res16x_1, res16x_2 = res16x.split(
            [(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1
        )
        res16x_1 = self.fusion4(res16x_1, feature_mem)
        res16x_2 = self.conv4(res16x_2)
        res16x = torch.cat((res16x_1, res16x_2), dim=1)

        res_dehaze = res16x
        in_ft = res16x * 2
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze
        res16x_1, res16x_2 = res16x.split(
            [(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1
        )
        feature_mem_up = [res16x_1]

        res16x = self.convd16x(res16x)
        res16x = F.upsample(res16x, res8x.size()[2:], mode="bilinear")
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x
        res8x_1, res8x_2 = res8x.split(
            [(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1
        )
        res8x_1 = self.fusion_4(res8x_1, feature_mem_up)
        res8x_2 = self.conv_4(res8x_2)
        feature_mem_up.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)

        res8x = self.convd8x(res8x)
        res8x = F.upsample(res8x, res4x.size()[2:], mode="bilinear")
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x
        res4x_1, res4x_2 = res4x.split(
            [(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1
        )
        res4x_1 = self.fusion_3(res4x_1, feature_mem_up)
        res4x_2 = self.conv_3(res4x_2)
        feature_mem_up.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)

        res4x = self.convd4x(res4x)
        res4x = F.upsample(res4x, res2x.size()[2:], mode="bilinear")
        res2x = torch.add(res4x, res2x)
        res2x = self.dense_2(res2x) + res2x - res4x
        res2x_1, res2x_2 = res2x.split(
            [(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1
        )
        res2x_1 = self.fusion_2(res2x_1, feature_mem_up)
        res2x_2 = self.conv_2(res2x_2)
        feature_mem_up.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)

        res2x = self.convd2x(res2x)
        res2x = F.upsample(res2x, x.size()[2:], mode="bilinear")
        x = torch.add(res2x, x)
        x = self.dense_1(x) + x - res2x
        x_1, x_2 = x.split([(x.size()[1] // 2), (x.size()[1] // 2)], dim=1)
        x_1 = self.fusion_1(x_1, feature_mem_up)
        x_2 = self.conv_1(x_2)
        x = torch.cat((x_1, x_2), dim=1)

        x = self.conv_output(x)

        return x


class MSBDNModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        args, _ = parser.parse_known_args()
        parser.add_argument(
            "--n_blocks", type=int, default=18, help="number of resblocks"
        )
        parser.add_argument(
            "--dest", action="store_true", help="if true uses Inner-DeST",
        )
        parser.add_argument(
            "--aug",
            action="store_true",
            help="if true uses augmentation during testing",
        )
        parser.set_defaults(epochs=100, lr=1e-4)
        return parser

    def __init__(self, args):
        BaseModel.__init__(self, args)
        self.loss_names = ["content", "final"]
        self.visual_names = ["lr", "sr", "hr"]

        self.model = MSBDN(args.n_blocks)

        if self.args.dest:
            self.loss_names.append("dest")
            self.dest = DeST(
                "54", 512, 512, inner=True, Q=25 * 4, K=384, mean_shift=True
            )

        self.configure_optimizers()

    def set_input(self, input):
        self.lr = input["lr"]
        self.hr = input["hr"]
        if self.args.gpu is not None:
            self.lr = self.lr.cuda(self.args.gpu, non_blocking=True)
            self.hr = self.hr.cuda(self.args.gpu, non_blocking=True)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), self.args.lr,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [50], 0.1)

    def forward(self, x):
        return self.model(x)

    def optimize_parameters(self, global_step):
        self.sr = self.model(self.lr)
        self.content_loss = F.mse_loss(self.sr, self.hr)
        self.final_loss = self.content_loss

        if self.args.dest:
            self.dest_loss = self.dest(self.sr, self.hr)
            self.final_loss += self.dest_loss

        self.optimizer.zero_grad()
        self.final_loss.backward()
        self.optimizer.step()
        return self.final_loss, self.sr.size(0)

    def eval(self):
        with torch.no_grad():
            self.sr = self.test(self.lr)
            torch.cuda.empty_cache()
        diff = self.sr - self.hr
        mse = diff.pow(2).mean()
        psnr = -10 * math.log10(mse)
        return psnr

    def pad_test(self, x):
        aug = self.args.aug

        padding = 64
        _, _, h, w = x.shape
        pad_w = padding - w % padding
        pad_h = padding - h % padding

        pad_x = F.pad(
            x,
            (pad_w - pad_w // 2, pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            mode="reflect",
        )

        if aug:
            im_augs = [
                pad_x,
                torch.flip(pad_x, [2]),
                torch.flip(pad_x, [3]),
                torch.flip(pad_x, [2, 3]),
            ]
            output_augs = []
            for im_aug in im_augs:
                with torch.no_grad():
                    output = self.forward(im_aug)
                    torch.cuda.empty_cache()
                output_augs.append(output)
            output_augs = [
                output_augs[0],
                torch.flip(output_augs[1], [2]),
                torch.flip(output_augs[2], [3]),
                torch.flip(output_augs[3], [2, 3]),
            ]
            output = torch.mean(torch.cat(output_augs, dim=0), axis=0, keepdim=True)
        else:
            with torch.no_grad():
                output = self.forward(pad_x)
                torch.cuda.empty_cache()

        output = output[
            :, :, pad_h // 2 : -(pad_h - pad_h // 2), pad_w // 2 : -(pad_w - pad_w // 2)
        ]
        return output

    def test(self, x):
        # This trick is designed for I-HAZE and O-HAZE only
        _, _, h, w = x.shape
        x_1 = x[:, :, :, : w // 4 * 2]
        x_2 = x[:, :, :, w // 4 * 1 : w // 4 * 3]
        x_3 = x[:, :, :, w // 4 * 2 :]

        x_1 = self.pad_test(x_1)
        x_2 = self.pad_test(x_2)
        x_3 = self.pad_test(x_3)

        x[:, :, :, : w // 4] = x_1[:, :, :, : w // 4]
        x[:, :, :, w // 4 : w // 4 * 2] = (
            x_1[:, :, :, w // 4 :] + x_2[:, :, :, : w // 4]
        ) / 2
        x[:, :, :, w // 4 * 2 : w // 4 * 3] = (
            x_2[:, :, :, w // 4 :] + x_3[:, :, :, : w // 4]
        ) / 2
        x[:, :, :, w // 4 * 3 :] = x_3[:, :, :, w // 4 :]

        return x
