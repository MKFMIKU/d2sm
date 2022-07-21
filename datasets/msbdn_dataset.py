# Follow the protocol of https://github.com/BookerDeWitt/MSBDN-DFF/blob/master/datasets/dataset_hf5.py
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


class MSBDNDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--data_root", metavar="DIR",
                            default="/tmp_data/running/MSBDN", help="path to dataset")
        parser.add_argument(
            "--rgb_range", type=int, default=1, help="maximum value of RGB"
        )
        return parser

    def __init__(self, args, training):
        BaseDataset.__init__(self, args, training)
        if self.training:
            self.hr_image_paths = CD.make_image_dataset(
                os.path.join(self.args.data_root, 'RESIDE_npy', 'label')
            )
            self.lr_image_paths = CD.make_image_dataset(
                os.path.join(self.args.data_root, 'RESIDE_npy', 'data')
            )
        else:
            self.hr_image_paths = CD.make_image_dataset(
                os.path.join(self.args.data_root, 'SOTS', 'HR')
            )
            self.lr_image_paths = CD.make_image_dataset(
                os.path.join(self.args.data_root, 'SOTS', 'HR_hazy')
            )

    def __getitem__(self, index):
        if self.training:
            HR_patch = np.load(self.hr_image_paths[index])
            LR_patch = np.load(self.lr_image_paths[index])

            LR_patch = np.clip(LR_patch, 0, 1)  # we might get out of bounds due to noise
            HR_patch = np.clip(HR_patch, 0, 1)  # we might get out of bounds due to noise
            LR_patch = np.asarray(LR_patch, np.float32)
            HR_patch = np.asarray(HR_patch, np.float32)

            flip_channel = random.randint(0, 1)
            if flip_channel != 0:
                LR_patch = np.flip(LR_patch, 2)
                HR_patch = np.flip(HR_patch, 2)
            rotation_degree = random.randint(0, 3)
            LR_patch = np.ascontiguousarray(np.rot90(LR_patch, rotation_degree, (1, 2)))
            HR_patch = np.ascontiguousarray(np.rot90(HR_patch, rotation_degree, (1, 2)))
            hr, lr = torch.from_numpy(HR_patch).float(), torch.from_numpy(LR_patch).float()
        else:
            hr = imageio.imread(self.hr_image_paths[index % len(self.hr_image_paths)])
            lr = imageio.imread(self.lr_image_paths[index % len(self.hr_image_paths)])
            hr, lr = CD.np2Tensor(hr, lr, rgb_range=self.args.rgb_range)
        return {"hr": hr, "lr": lr}

    def __len__(self):
        return len(self.hr_image_paths)
