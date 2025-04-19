__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
from torch import optim
from torch import nn
from typing import AnyStr, Optional, List, Dict, Any, Self
from dataset import DatasetException
import logging

from dl import TrainingException

logger = logging.getLogger('dl.HyperParams')

"""
    Generic class to encapsulate feature_name configuration parameters of training of any Neural Networks. 
    Not all these parameters are to be tuned during training. The label reshaping function is
    a not tunable parameters used to width label input_tensor if needed
"""


class HyperParams(object):
    optim_adam_label: AnyStr = 'adam'
    optim_nesterov_label: AnyStr = 'nesterov'
    default_train_eval_ratio = 0.85

    def __init__(self,
                 lr: float,
                 momentum: float,
                 epochs: int,
                 optim_label: AnyStr,
                 batch_size: int,
                 loss_function: nn.Module,
                 drop_out: float,
                 train_eval_ratio: float = default_train_eval_ratio,
                 encoding_len: int = -1,
                 weight_initialization: Optional[AnyStr] = 'xavier',
                 class_weights: Optional[torch.Tensor] = None):
        """
            Constructor
            @param lr: Learning rate for the selected optimizer
            @type lr: float
            @param momentum: Momentum rate for the selected optimizer
            @type momentum: float
            @param epochs: Number of epochs or iterations
            @type epochs: int
            @param optim_label: Label for the optimizer (i.e. 'sgd', 'adam', ...)
            @type optim_label: str
            @param batch_size: Size of the batch used for the batch normalization
            @type batch_size: int
            @param loss_function: PyTorch nn.Module loss function (i.e BCELoss, MSELoss....)
            @type loss_function: torch.nn.Module
        """
        HyperParams.__check_constructor(lr, momentum, epochs, batch_size, train_eval_ratio, drop_out)
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.momentum = momentum
        self.encoding_len = encoding_len
        self.train_eval_ratio = train_eval_ratio
        self.weight_initialization = weight_initialization
        self.optim_label = optim_label
        self.drop_out = drop_out
        self.class_weights = class_weights
        torch.manual_seed(42)

    @classmethod
    def build(cls, attributes: Dict[AnyStr, Any]) -> Self:
        """
        Alternative constructor that relies on dictionary of attributes
        @param attributes: Dictionary of attributes of hyper-parameters
        @type attributes: Dict[AnyStr, Any]
        @return: Instance of this class, HyperParam
        @rtype: HyperParam
        """
        assert len(attributes), 'Attributes for hyper parameters are undefined'

        learning_rate = attributes.get('learning_rate', 0.001)
        epochs = attributes.get('epochs', 24)
        batch_size = attributes.get('batch_size', 64)
        loss_function = attributes.get('loss_function', nn.CrossEntropyLoss())
        momentum = attributes.get('momentum', 0.98)
        encoding_len = attributes.get('encoding_len', -1)
        train_eval_ratio = attributes.get('train_eval_ratio', 0.9)
        weight_initialization = attributes.get('weight_initialization', 'xavier')
        optim_label = attributes.get('optim_label', 'adam')
        drop_out = attributes.get('drop_out', 0.0)
        class_weights = attributes['class_weights'] if 'class_weights' in attributes else None

        return cls(learning_rate,
                   momentum,
                   epochs,
                   optim_label,
                   batch_size,
                   loss_function,
                   drop_out,
                   train_eval_ratio,
                   encoding_len,
                   weight_initialization,
                   class_weights)

    @staticmethod
    def test_conversion(label_conversion_func) -> torch.Tensor:
        x = torch.rand(20)
        return label_conversion_func(x)

    def initialize_weight(self, modules: List[nn.Module]) -> None:
        """
        In-place initialization weight of a list of linear module given an encoder model
        @param modules: torch module to be initializes
        @type modules: List
        """
        import torch_geometric

        def is_layer_module(m: nn.Module) -> bool:
            return isinstance(m, (nn.Linear,
                                  nn.Conv2d,
                                  nn.Conv1d,
                                  torch_geometric.nn.GraphConv,
                                  torch_geometric.nn.GCNConv,
                                  torch_geometric.nn.Linear)
                              )

        match self.weight_initialization:
            case 'normal':
                [nn.init.normal_(module.weight) for module in modules if is_layer_module(module)]
            case 'xavier':
                [nn.init.xavier_uniform_(module.weight) for module in modules if is_layer_module(module)]
            case 'constant':
                [nn.init.constant_(module.weight, val=0.5) for module in modules if is_layer_module(module)]
            case _:
                raise TrainingException(f'initialization {self.weight_initialization} '
                                        'for layer module weights is not supported')
        [nn.init.constant_(module.bias, val=0.1) for module in modules if is_layer_module(module)]

    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Select the optimizer for generated from encoder_model parameters given the optimization label
            - SGD with Nesterov momentum
            - Adam
            - Plain vanilla SGD
        @param model: Neural Network model
        @type model: NeuralModel
        @return: Appropriate Torch optimizer
        @rtype: torch.optim.Optimizer
        """
        match self.optim_label:
            case HyperParams.optim_adam_label:
                optimizer = optim.Adam(params=model.parameters(),
                                       lr=self.learning_rate,
                                       betas=(self.momentum, 0.998))
            case HyperParams.optim_nesterov_label:
                optimizer = optim.SGD(model.parameters(),
                                      lr=self.learning_rate,
                                      momentum=self.momentum,
                                      nesterov=True)
            case _:
                logger.warn(f'Type of optimization {self.optim_label} not supported: reverted to SGD')
                optimizer = optim.SGD(model.parameters(),
                                      lr=self.learning_rate,
                                      momentum=self.momentum,
                                      nesterov=False)
        # Set the gradient values of the selected optimizer to 0.0
        optimizer.zero_grad()
        return optimizer

    def __repr__(self) -> str:
        return f'   Learning rate:      {self.learning_rate}\n   Momentum:           {self.momentum}' \
               f'\n   Number of epochs:   {self.epochs}\n   Batch size:         {self.batch_size}'\
               f'\n   Optimizer:          {self.optim_label}\n   Loss function:      {repr(self.loss_function)}'\
               f'\n   Drop out rate:      {self.drop_out}\n   Train split ratio:  {self.train_eval_ratio}'

    def get_label(self) -> str:
        return f'lr.{self.learning_rate}-bs.{self.batch_size}'

    def grid_search(self, lr_rates: List[float], batch_sizes: List[int]) -> iter:
        """
            Generate multiple Hyper-parameter using a list of learning rate and batch-size
            @param lr_rates:  List of learning rates
            @type lr_rates: list of float
            @param batch_sizes: List of batch sizes
            @type batch_sizes: list of batch size
            @return: Iterator for Hyper-params
            @rtype: iter
        """
        if not HyperParams.__check_grid_search(lr_rates, batch_sizes):
            raise DatasetException('Some parameters for the grid search are out-of-bound')

        return (HyperParams(
            lr,
            self.momentum,
            self.epochs,
            self.optim_label,
            batch_size,
            self.loss_function,
            self.drop_out,
            self.weight_initialization) for lr in lr_rates for batch_size in batch_sizes)

    # ---------------------  Helper private methods -----------------------
    @staticmethod
    def __check_constructor(lr: float,
                            momentum: float,
                            epochs: int,
                            batch_size: int,
                            train_eval_ratio: float,
                            drop_out: float) -> None:
        assert 1e-6 <= lr <= 0.1, f'Learning rate {lr} should be [1e-6, 0.1]'
        assert 2 <= epochs <= 2048, f'Number of epochs {epochs} should be [3, 2048]'
        assert 0.4 <= momentum <= 0.999, f'Context stride {momentum} should be [0.5, 0.999]'
        assert 1 <= batch_size <= 2048, f'Size of batch {batch_size} should be [2, 1024]'
        assert 0.5 < train_eval_ratio < 0.98, f'Train eval ratio {train_eval_ratio} is out of range ]0.5, 9.98['
        assert 0.0 <= drop_out <= 1.0, f'Drop out {drop_out} should be [0, 1]'

    @staticmethod
    def __check_grid_search(lr_rates: List[float], batch_sizes: List[int]) -> bool:
        accepted = True if len(lr_rates) > 0 and len(batch_sizes) > 1 else False
        return accepted and \
            all(0.0 < lr < 0.1 for lr in lr_rates) and \
            all(0 < batch_size <= 256 for batch_size in batch_sizes)

