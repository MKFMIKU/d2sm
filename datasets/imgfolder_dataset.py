import os
import imageio
from datasets.base_dataset import BaseDataset
import datasets.common_dataset as CD


class ImgFolderDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--data_root", metavar="DIR", help="path to dataset")
        parser.add_argument(
            "--rgb_range", type=int, default=1, help="maximum value of RGB"
        )
        return parser

    def __init__(self, args, training):
        BaseDataset.__init__(self, args, training)
        self.image_paths = CD.make_image_dataset(self.args.data_root)

    def __getitem__(self, index):
        img = imageio.imread(self.image_paths[index])
        img = CD.np2Tensor(img, rgb_range=self.args.rgb_range)[0]

        return {"img": img, "path": self.image_paths[index]}

    def __len__(self):
        return len(self.image_paths)
