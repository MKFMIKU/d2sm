from abc import ABC, abstractmethod
import torch.utils.data as data


class BaseDataset(data.Dataset, ABC):
    def __init__(self, args, training=True):
        self.args = args
        self.training = training

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass
