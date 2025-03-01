__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import AnyStr, Self, List, Dict, Any
from dl.block.conv import ConvDataType
from torch import Tensor
from dl import DLException
import logging


logger = logging.getLogger('dl.model.NeuralModel')


__all__ = ['NeuralModel']

"""
Abstract base class for Neural network models. The sub-classes have to implement get_model,
forward and save methods
"""


class NeuralModel(torch.nn.Module, ABC):
    def __init__(self, model_id: AnyStr, modules_seq: nn.Module) -> None:
        """
        Constructor
        @param model_id: Identifier for this model
        @type model_id: str
        @param modules_seq: Model as a Torch neural module
        @type modules_seq: nn.Module
        """
        super(NeuralModel, self).__init__()
        self.model_id = model_id
        self.modules_seq = modules_seq

    def add_noise(self, x: Tensor) -> Tensor:
        """
        Add noise to the input data if defined in the constructor
        @param x: input training data as tensor
        @type x: torch.Tensor
        @return: Noisy tensor if noise_func has been defined, the argument tensor otherwise
        @rtype: torch.Tensor
        """
        return self.noise_func(x) if self.noise_func is not None else x

    def get_modules(self) -> List[nn.Module]:
        return list(self.modules_seq.children())

    def list_modules(self, index: int = 0) -> AnyStr:
        modules = [f'{idx+index}: {str(module)}' for idx, module in enumerate(self.get_modules())]
        return '\n'.join(modules)

    def get_flatten_output_size(self) -> ConvDataType:
        raise DLException('Abstract class cannot have a flatten output size')

    def forward(self, x: Tensor) -> Tensor:
        """
        Execute the default forward method for all neural network models inherited from this class
        @param x: Input tensor
        @type x: Torch tensor
        @return: Prediction for the input
        @rtype: Torch tensor
        """
        # print(f'Input {self.model_id}\n{x.shape}')
        x = self.modules_seq(x)
        # print(f'Output {self.model_id}\n{x.shape}')
        return x

    def get_in_features(self) -> int:
        raise NotImplementedError('NeuralModel.get_in_features undefined for abstract neural model')

    def get_out_features(self) -> int:
        """
        Polymorphic method to retrieve the number of output features
        @return: Number of input features
        @rtype: int
        """
        raise NotImplementedError('NeuralModel.get_out_features undefined for abstract neural model')

    def get_latent_features(self) -> int:
        raise NotImplementedError('NeuralModel.get_latent_features undefined for abstract neural model')

    def transpose(self, extra: nn.Module = None) -> Self:
        raise NotImplementedError('NeuralModel.invert is an abstract method')

    def __str__(self) -> AnyStr:
        return f'\n{self.model_id}\n{str(self.modules_seq)}'

    def __repr__(self) -> AnyStr:
        return '\n'.join([f'{idx}: {str(module)}' for idx, module in enumerate(self.get_modules())])

    def save(self, extra_params: dict = None):
        raise NotImplementedError('NeuralModel.save is an abstract method')



class NeuralBuilder(ABC):
    def __init__(self, model_id: AnyStr, keys: List[AnyStr]) -> None:
        self.__attributes = dict.fromkeys(keys)
        self.__attributes['model_id'] = model_id

    def set(self, key: AnyStr, value: Any) -> Self:
        self.__attributes[key] = value
        return self

    @abstractmethod
    def build(self) -> Any:
        raise DLException('Neural Builder is an abstract class')
