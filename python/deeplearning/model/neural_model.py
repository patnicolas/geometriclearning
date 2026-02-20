__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard Library imports
from abc import ABC, abstractmethod
from typing import AnyStr, Self, List, Dict, Any
import logging
# 3rd Party imports
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
# Library imports
from deeplearning import MLPException
from deeplearning.block.conv import ConvDataType
from deeplearning.block.neural_block import NeuralBlock
import python
__all__ = ['NeuralModel', 'NeuralBuilder']

from deeplearning.training.neural_training import NeuralTraining


class NeuralModel(torch.nn.Module, ABC):
    """
    Abstract base class for Neural network models. The constructors of the sub-classes needs
    to define the sequence of neural blocks.
    """
    def __init__(self, model_id: AnyStr) -> None:
        """
        Constructor
        @param model_id: Identifier for this model
        @type model_id: str
        """
        super(NeuralModel, self).__init__()
        self.model_id = model_id
        self.modules_seq = None

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
        return list(self.modules_seq.children()) if self.modules_seq is not None else []

    def show_modules(self, index: int = 0) -> AnyStr:
        modules = [f'{idx+index}: {str(module)}' for idx, module in enumerate(self.get_modules())]
        return '\n'.join(modules)

    def get_flatten_output_size(self) -> ConvDataType:
        raise MLPException('Abstract class cannot have a flatten output size')

    def train_model(self,
                    training: NeuralTraining,
                    train_loader: DataLoader,
                    val_loader: DataLoader) -> None:
        """
        Training and evaluation of models using Neural Training and train loader for training and evaluation data

        @param training: Wrapper class for training Neural Network
        @type training: NeuralTraining
        @param train_loader: Loader for the training data set
        @type train_loader: torch.utils.data.DataLoader
        @param val_loader:   Loader for the validation data set
        @type val_loader:  torch.utils.data.DataLoader
        """
        pass

    def forward(self, x: Tensor) -> Tensor:
        """
        Execute the default forward method for all neural network models inherited from this class
        @param x: Input tensor
        @type x: Torch tensor
        @return: Prediction for the input
        @rtype: Torch tensor
        """
        logging.debug(f'{self.model_id=}\n{x.shape=}')
        x = self.modules_seq(x)
        logging.debug(f'Output {self.model_id=}\n{x.shape=}')
        return x

    def _register_modules(self, conv_blocks: List[NeuralBlock], fully_connected: List[NeuralBlock]) -> None:
        if self.modules_seq is None:
            modules_list = [module for block in conv_blocks for module in block.modules_list]
            # If fully connected are provided
            if fully_connected is not None:
                # Flatten the output of the last convolution block/layer
                modules_list.append(nn.Flatten())
                # Generate the list of modules for all the multi-layer perceptron blocks
                [modules_list.append(module) for block in fully_connected for module in block.modules_list]
            self.modules_seq = nn.Sequential(*modules_list)

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
    """
    A builder for any Neural Network
    """
    def __init__(self, model_attributes: Dict[AnyStr, Any]) -> None:
        """
        Constructor for this Builder
        @param model_attributes:  Dictionary of model configuration parameters
        @type model_attributes: Dictionary
        """
        self.model_attributes = model_attributes

    def set(self, key: AnyStr, value: Any) -> Self:
        """
        Add/update dynamically the torch module as value of attributes dict.
        @param key: Key or name of the configuration parameter
        @type key:  str
        @param value: Value for the configuration parameter
        @type value: Any
        @return: Instance of this builder
        @rtype: NeuralBuilder
        """
        self.model_attributes[key] = value
        return self

    @abstractmethod
    def build(self) -> NeuralModel:
        """
        A Neural Builder is an abstract class
        """
        pass
