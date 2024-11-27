__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch
import torch.nn as nn
from abc import ABC
from typing import AnyStr, Self, List, Callable
from dl import ConvException, DLException
from dl.training.neural_net_training import NeuralNetTraining
from dl.training.hyper_params import HyperParams
from torch.utils.data import DataLoader
from dl.training.exec_config import ExecConfig
import logging
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute the default forward method for all neural network models inherited from this class
        @param x: Input tensor
        @type x: Torch tensor
        @return: Prediction for the input
        @rtype: Torch tensor
        """
        print(f'Input {self.model_id}\n{x.shape}')
        x = self.model(x)
        print(f'Output {self.model_id}\n{x.shape}')
        return x

    def do_train(self,
                 loaders: (DataLoader, DataLoader),
                 hyper_parameters: HyperParams,
                 metric_labels: List[AnyStr],
                 exec_config: ExecConfig,
                 plot_title: AnyStr) -> None:
        """
        Execute the training, evaluation and metrics for any model for MNIST data set
        @param loaders: Tuple/Pair of loader for training and evaluation data
        @type loaders: Tuple[DataLoader, DataLoader]
        @param hyper_parameters: Hyper-parameters for the execution of the
        @type hyper_parameters: HyperParams
        @param metric_labels: List of metrics to be used
        @type metric_labels: List
        @param exec_config: Configuration for the execution of training set
        @type exec_config: ExecConfig
        @param plot_title: Labeling metric for output to file and plots
        @type plot_title: str
        """
        try:
            network = NeuralNetTraining.build(self, hyper_parameters, metric_labels, exec_config)
            plot_title = f'{self.model_id}_metrics_{plot_title}'
            network(plot_title=plot_title, loaders=loaders)
        except ConvException as e:
            logger.error(str(e))
            raise DLException(e)
        except AssertionError as e:
            logger.error(str(e))
            raise DLException(e)

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

    def invert(self) -> Self:
        raise NotImplementedError('NeuralModel.invert is an abstract method')

    def __repr__(self) -> AnyStr:
        return f'Model: {self.model_id}'

    def get_model(self) -> torch.nn.Module:
        return self.model

    def save(self, extra_params: dict = None):
        raise NotImplementedError('NeuralModel.save is an abstract method')



