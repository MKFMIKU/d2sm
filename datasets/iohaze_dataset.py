from datasets.base_dataset import BaseDataset
import datasets.common_dataset as CD

import torch
import torch.utils.data as data
import torchvision.transforms as tf
import numpy as np
import random
import os
import glob
import imageio


class IOHAZEDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument(
            "--data_root",
            metavar="DIR",
            default="/tmp_data/running/MSBDN",
            help="path to dataset",
        )
        parser.add_argument(
            "--patch_size", default=512, type=int, help="patch size of high-resolution"
        )
        parser.add_argument(
            "--rgb_range", type=int, default=1, help="maximum value of RGB"
        )
        return parser

    def __init__(self, args, training):
        BaseDataset.__init__(self, args, training)
        self.hr_image_paths = CD.make_image_dataset(
            os.path.join(self.args.data_root, "train" if self.training else "val", "GT")
        )
        self.lr_image_paths = CD.make_image_dataset(
            os.path.join(
                self.args.data_root, "train" if self.training else "val", "hazy"
            )
        )
        self.hr_image_paths.sort(key=lambda x: int(os.path.basename(x)[:2]))
        self.lr_image_paths.sort(key=lambda x: int(os.path.basename(x)[:2]))

    def __getitem__(self, index):
        if self.training:
            lr = imageio.imread(self.lr_image_paths[index % len(self.lr_image_paths)])
            hr = imageio.imread(self.hr_image_paths[index % len(self.lr_image_paths)])

            lr, hr = CD.get_patch(lr, hr, patch_size=self.args.patch_size, scale=1)
            lr, hr = CD.augment(lr, hr)
            lr, hr = CD.np2Tensor(lr, hr, rgb_range=self.args.rgb_range)
            return {"lr": lr, "hr": hr}
        else:
            lr = imageio.imread(self.lr_image_paths[index])
            hr = imageio.imread(self.hr_image_paths[index])
            lr, hr = CD.np2Tensor(lr, hr, rgb_range=self.args.rgb_range)
            return {"lr": lr, "hr": hr, "path": self.lr_image_paths[index]}

    def __len__(self):
        return len(self.lr_image_paths) * (1000 if self.training else 1)
