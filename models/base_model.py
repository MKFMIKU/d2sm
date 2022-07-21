from collections import OrderedDict
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    @staticmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self, input):
        pass

    @staticmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def optimize_parameters(self, global_step):
        pass

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, name + "_loss")
                )  # float(...) works for both scalar tensor and float number
        return errors_ret

    def print_networks(self, verbose=False):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print("---------- Networks initialized -------------")
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
            if verbose:
                print(self)
        print(
            "[Network %s] Total number of parameters : %.3f M"
            % (self.args.model, num_params / 1e6)
        )
        print("-----------------------------------------------")
