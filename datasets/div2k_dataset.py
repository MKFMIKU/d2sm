# Follow the protocol of https://github.com/thstkdgus35/EDSR-PyTorch/blob/master/src/data/div2k.py
import os
import imageio
from datasets.base_dataset import BaseDataset
import datasets.common_dataset as CD


class DIV2KDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--data_root", metavar="DIR", help="path to dataset")
        parser.add_argument(
            "--patch_size", default=192, type=int, help="patch size of high-resolution"
        )
        parser.add_argument(
            "--rgb_range", type=int, default=255, help="maximum value of RGB"
        )
        parser.set_defaults(input_nc=3, output_nc=3)

        args, _ = parser.parse_known_args()
        if "scale" not in args:
            parser.add_argument(
                "--scale", type=int, default=4, help="super resolution scale"
            )
        return parser

    def __init__(self, args, training):
        BaseDataset.__init__(self, args, training)
        self.lr_image_paths = CD.make_image_dataset(
            os.path.join(
                self.args.data_root,
                "DIV2K_train_LR_bicubic",
                "X%d_sub" % self.args.scale,
            )
        )
        self.hr_image_paths = CD.make_image_dataset(
            os.path.join(self.args.data_root, "DIV2K_train_HR_sub")
        )

    def __getitem__(self, index):
        lr = imageio.imread(self.lr_image_paths[index])
        hr = imageio.imread(self.hr_image_paths[index])

        lr, hr = CD.get_patch(
            lr, hr, patch_size=self.args.patch_size, scale=self.args.scale
        )
        lr, hr = CD.augment(lr, hr)
        lr, hr = CD.np2Tensor(lr, hr, rgb_range=self.args.rgb_range)

        return {"lr": lr, "hr": hr}

    def __len__(self):
        return len(self.hr_image_paths)
