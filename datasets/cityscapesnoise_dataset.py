from datasets.base_dataset import BaseDataset
import datasets.common_dataset as CD
import numpy as np
import imageio
import torch
import torchvision.transforms.functional as TF
import os

class CITYSCAPESNOISEDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--data_root", metavar="DIR", help="path to dataset")
        parser.add_argument(
            "--patch_size", default=192, type=int, help="patch size of high-resolution"
        )
        parser.add_argument(
            "--rgb_range", type=int, default=1, help="maximum value of RGB"
        )
        parser.set_defaults(input_nc=3, output_nc=3)
        return parser

    def __init__(self, args, training):
        BaseDataset.__init__(self, args, training)
        self.image_paths = CD.make_image_dataset(os.path.join(self.args.data_root, 'train' if self.training else 'val'))

    def __getitem__(self, index):
        hr = imageio.imread(self.image_paths[index % len(self.image_paths)])
        H, W, _ = hr.shape

        if self.training:
            hr = CD.get_patch(hr, patch_size=self.args.patch_size, scale=1)[0]
            hr = CD.augment(hr)[0]
            hr = CD.np2Tensor(hr, rgb_range=self.args.rgb_range)[0]
            noisy_hr = hr.clone()
            noise_level = (
                torch.FloatTensor([np.random.uniform(0, self.args.sigma)]) / 255.0
            )
            noise = torch.randn(hr.size()).mul_(noise_level).float()
            noisy_hr.add_(noise)
        else:
            eval_sigma = 25
            hr = CD.np2Tensor(hr, rgb_range=self.args.rgb_range)[0]
            # hr = TF.resize(hr, [hr.size(1) // 2, hr.size(2) // 2])  # For quick evaluation
            C, H, W = hr.shape
            np.random.seed(seed=0)
            noise = np.random.normal(0, eval_sigma / 255.0, (H, W, C)).transpose(
                (2, 0, 1)
            )
            noisy_hr = hr + torch.FloatTensor(noise)
            noise_level = torch.FloatTensor([eval_sigma / 255.0])

        noise_level = noise_level.unsqueeze(1).unsqueeze(1)

        return {"noisy_hr": noisy_hr, "hr": hr, "noise_level": noise_level}

    def __len__(self):
        return len(self.image_paths)
