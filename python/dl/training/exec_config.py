__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from dataclasses import dataclass
import torch
from typing import AnyStr, NoReturn


class ExecConfig(object):
    def __init__(self,
                 empty_cache: bool,
                 mix_precision: bool,
                 subset_size: int,
                 device_config: AnyStr = None,
                 pin_mem: bool = True):
        """
        Constructor for the configuration of the execution of training of DL models
        @param empty_cache: Flat to empty cache between epochs
        @type empty_cache: bool
        @param mix_precision: Flag to support upport mix precision Float16 for data and Float32 for model
        @type mix_precision: bool
        @param subset_size: Select a subset of the training data is >0, or entire data set if -1
        @type subset_size: int
        @param device_config: Device {'cpu', 'cuda' ..}
        @type device_config: str
        @param pin_mem: Flag to enable pin memory for data set loader
        @type pin_mem: bool
        """
        self.empty_cache = empty_cache
        self.mix_precision = mix_precision
        self.subset_size = subset_size
        self.device_config = device_config
        self.pin_mem = pin_mem

    def __str__(self) -> AnyStr:
        return (f'\nEmpty cache: {self.empty_cache}\nMix precision {self.mix_precision}\nSubset size: {self.subset_size}'
                f'\nDevice: {self.device_config}\nPin memory: {self.pin_mem}')

    def apply_mix_precision(self, x: torch.Tensor) -> torch.Tensor:
        return x.half() if self.mix_precision else x

    def apply_empty_cache(self) -> NoReturn:
        if self.empty_cache:
            torch.mps.empty_cache()

    def get_device(self) -> (AnyStr, torch.device):
        """
        Retrieve the device (CPU, GPU) used for the execution, training and evaluation of this Neural network
        @return: Pair (device name, torch device)
        @rtype: Tuple[AnyStr, torch.device]
        """
        assert self.device_config is None or self.device_config in ['auto', 'cpu', 'mps', 'cuda'], \
            f'Device {self.device_config} is not supported'

        if self.device_config is None:
            if torch.cuda.is_available():
                print("Using CUDA GPU")
                return 'cuda', torch.device("cuda")
            elif torch.backends.mps.is_available():
                print("Using MPS GPU")
                return 'mps', torch.device("mps")
            else:
                print("Using CPU")
                return 'cpu', torch.device("cpu")
        else:
            print(f'Using {self.device_config}')
            return self.device_config, torch.device(self.device_config)


