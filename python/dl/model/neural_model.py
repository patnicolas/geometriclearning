__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
import torch.nn as nn
from abc import ABC
from typing import AnyStr, Self, List, Callable
from torch.utils.data import DataLoader
from dl import DLException
import logging

from dl.training.neural_net_training import NeuralNetTraining

logger = logging.getLogger('dl.model.NeuralModel')


__all__ = ['NeuralModel']

"""
Abstract base class for Neural network models. The sub-classes have to implement get_model,
forward and save methods
"""


class NeuralModel(torch.nn.Module, ABC):
    def __init__(self,
                 model_id: AnyStr,
                 model: torch.nn.Module,
                 noise_func: Callable[[torch.Tensor], torch.Tensor] = None) -> None:
        """
        Constructor
        @param model_id: Identifier for this model
        @type model_id: str
        @param model: Model as a Torch neural module
        @type model: nn.Module
        @param noise_func: Function to add noise to input data (features)
        @param noise_func: Callable (noise_factor, input)
        """
        super(NeuralModel, self).__init__()
        self.model_id = model_id
        self.model = model
        self.noise_func = noise_func

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the input data if defined in the constructor
        @param x: input training data as tensor
        @type x: torch.Tensor
        @return: Noisy tensor if noise_func has been defined, the argument tensor otherwise
        @rtype: torch.Tensor
        """
        return self.noise_func(x) if self.noise_func is not None else x

    def get_modules(self) -> List[nn.Module]:
        return list(self.model.children())

    def list_modules(self, index: int = 0) -> AnyStr:
        raise DLException('Cannot list module of abstract Neural model')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute the default forward method for all neural network models inherited from this class
        @param x: Input tensor
        @type x: Torch tensor
        @return: Prediction for the input
        @rtype: Torch tensor
        """
        print(f'Input {self.model_id}\n{x.shape}')
        # x= x.to(self.execution.target_device)
        x = self.model(x)
        print(f'Output {self.model_id}\n{x.shape}')
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
        return f'\n{self.model_id}\n{str(self.model)}'

    def __repr__(self) -> AnyStr:
        return '\n'.join([f'{idx}: {str(module)}' for idx, module in enumerate(self.get_modules())])

    def save(self, extra_params: dict = None):
        raise NotImplementedError('NeuralModel.save is an abstract method')



