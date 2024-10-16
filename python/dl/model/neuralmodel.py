__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch
import torch.nn as nn
from abc import abstractmethod
from typing import AnyStr, Self, List
from util import log_size

__all__ = ['NeuralModel']

"""
Abstract base class for Neural network models. The sub-classes have to implement get_model,
forward and save methods
"""


class NeuralModel(torch.nn.Module):
    def __init__(self, model_id: AnyStr, model: torch.nn.Module):
        """
        Constructor
        @param model_id: Identifier for this model
        @type model_id: str
        @param model: Model as a Torch neural module
        @type model: nn.Module
        """
        super(NeuralModel, self).__init__()
        self.model_id = model_id
        self.model = model

    def get_modules(self) -> List[nn.Module]:
        return list(self.model.children())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute the default forward method for all neural network models inherited from this class
        @param x: Input tensor
        @type x: Torch tensor
        @return: Prediction for the input
        @rtype: Torch tensor
        """
        log_size(x, f'Input {self.model_id}')
        x = self.model(x)
        log_size(x, f'Output {self.model_id}')
        return x

    @abstractmethod
    def get_in_features(self) -> int:
        raise NotImplementedError('NeuralModel.get_in_features undefined for abstract neural model')

    @abstractmethod
    def get_out_features(self) -> int:
        """
        Polymorphic method to retrieve the number of output features
        @return: Number of input features
        @rtype: int
        """
        raise NotImplementedError('NeuralModel.get_out_features undefined for abstract neural model')

    @abstractmethod
    def get_latent_features(self) -> int:
        raise NotImplementedError('NeuralModel.get_latent_features undefined for abstract neural model')

    @abstractmethod
    def invert(self) -> Self:
        raise NotImplementedError('NeuralModel.invert is an abstract method')

    def __repr__(self) -> AnyStr:
        return f'Model: {self.model_id}'

    def get_model(self) -> torch.nn.Module:
        return self.model

    @abstractmethod
    def save(self, extra_params: dict = None):
        raise NotImplementedError('NeuralModel.save is an abstract method')

