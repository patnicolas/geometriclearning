__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

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
from typing import AnyStr, Optional, List, Dict, Any, Self
import logging
# 3rd Party imports
import torch
from torch import optim
from torch import nn
# Library imports
from dataset import DatasetException
from deeplearning.training import TrainingException
import python
__all__ = ['HyperParams']


class HyperParams(object):
    """
        Generic class to encapsulate feature_name configuration parameters of training of any Neural Networks.
        Not all these parameters are to be tuned during training. The label reshaping function is
        a not tunable parameters used to width label input_tensor if needed
    """

    optim_adam_label: AnyStr = 'adam'
    optim_nesterov_label: AnyStr = 'nesterov'
    default_train_eval_ratio = 0.85

    def __init__(self,
                 lr: float,
                 momentum: float,
                 weight_decay: float,
                 epochs: int,
                 optim_label: AnyStr,
                 batch_size: int,
                 loss_function: nn.Module,
                 drop_out: float,
                 train_eval_ratio: float = default_train_eval_ratio,
                 encoding_len: int = -1,
                 weight_initialization: Optional[AnyStr] = 'xavier',
                 class_weights: Optional[torch.Tensor] = None,
                 tensor_mix_precision: Optional[torch.dtype] = None,
                 checkpoint_enabled: bool = False,
                 target_device: Optional[AnyStr] = None):
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
            @param checkpoint_enabled: Flag to enable checkpoint output of activation
            @type checkpoint_enabled bool
        """
        HyperParams.__check_constructor(lr, momentum, epochs, batch_size, train_eval_ratio, drop_out)

        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.encoding_len = encoding_len
        self.train_eval_ratio = train_eval_ratio
        self.weight_initialization = weight_initialization
        self.optim_label = optim_label
        self.drop_out = drop_out
        self.class_weights = class_weights
        self.tensor_mix_precision = tensor_mix_precision
        self.checkpoint_enabled = checkpoint_enabled
        self.target_device = target_device
        torch.manual_seed(42)

    @classmethod
    def build(cls, attributes: Dict[AnyStr, Any]) -> Self:
        """
        Alternative constructor that relies on dictionary of attributes
        @param attributes: Dictionary of attributes of hyper-parameters
        @type attributes: Dict[AnyStr, Any]
        @return: Instance of this class, HyperParams
        @rtype: HyperParams
        """
        assert len(attributes), 'Attributes for hyper parameters are undefined'

        try:
            return cls(lr=attributes['learning_rate'],
                       momentum=attributes['momentum'],
                       weight_decay=attributes['weight_decay'],
                       epochs=attributes['epochs'],
                       optim_label=attributes['optim_label'],
                       batch_size=attributes['batch_size'],
                       loss_function=attributes.get('loss_function', nn.CrossEntropyLoss()),
                       drop_out=attributes['drop_out'],
                       train_eval_ratio=attributes['train_eval_ratio'],
                       encoding_len=attributes['encoding_len'],
                       weight_initialization=attributes['weight_initialization'],
                       class_weights=attributes.get('class_weights', None),
                       tensor_mix_precision=attributes.get('tensor_mix_precision', None),
                       checkpoint_enabled=attributes.get('checkpoint_enabled', False),
                       target_device=attributes.get('target_device', 'cpu'))
        except KeyError as e:
            logging.error(e)
            raise TrainingException(e)

    @staticmethod
    def test_conversion(label_conversion_func) -> torch.Tensor:
        x = torch.rand(20)
        return label_conversion_func(x)

    def initialize_weight(self, modules: List[nn.Module]) -> None:
        """
        In-place initialization weight of a list of linear module given an encoder model
        @param modules: All modules for which some need to be initialized
        @type modules: List[nn.Module]
        """
        import torch_geometric

        def is_layer_module(m: nn.Module) -> bool:
            return isinstance(m, (nn.Linear,
                                  nn.Conv2d,
                                  nn.Conv1d,
                                  torch_geometric.nn.GraphConv,
                                  torch_geometric.nn.SAGEConv,
                                  torch_geometric.nn.GCNConv,
                                  torch_geometric.nn.Linear)
                              )

        match self.weight_initialization:
            case 'kaiming':
                logging.info('Kaiming weights initialization')
                [nn.init.kaiming_uniform_(tensor=module.weight, mode='fan_in', nonlinearity='relu')
                 for module in modules if is_layer_module(module)]
            case 'normal':
                logging.info('Gaussian weights initialization')
                [nn.init.normal_(module.weight) for module in modules if is_layer_module(module)]
            case 'xavier':
                logging.info('Xavier weights initialization')
                [nn.init.xavier_uniform_(module.weight) for module in modules if is_layer_module(module)]
            case 'constant':
                logging.info('Constant weights initialization')
                [nn.init.constant_(module.weight, val=0.5) for module in modules if is_layer_module(module)]
            case _:
                raise TrainingException(f'initialization {self.weight_initialization} '
                                        'for layer module weights is not supported')

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
                                       weight_decay=self.weight_decay,
                                       betas=(self.momentum, 0.998))
            case HyperParams.optim_nesterov_label:
                optimizer = optim.SGD(model.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=5e-4,
                                      momentum=self.momentum,
                                      nesterov=True)
            case _:
                logging.warning(f'Type of optimization {self.optim_label} not supported: reverted to SGD')
                optimizer = optim.SGD(model.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=5e-4,
                                      momentum=self.momentum,
                                      nesterov=False)
        return optimizer

    def __repr__(self) -> str:
        device = self.target_device if self.target_device is not None else 'Auto'
        mix_precision = self.tensor_mix_precision if self.tensor_mix_precision is not None else 'No'

        return f'\n   Learning rate:      {self.learning_rate}\n   Momentum:           {self.momentum}' \
               f'\n   Number of epochs:   {self.epochs}\n   Batch size:         {self.batch_size}'\
               f'\n   Optimizer:          {self.optim_label}\n   Loss function:      {repr(self.loss_function)}'\
               f'\n   Drop out rate:      {self.drop_out}\n   Train split ratio:  {self.train_eval_ratio}' \
               f'\n   Device:             {device}\n   Checkpoint Enabled: {self.checkpoint_enabled }' \
               f'\n   Class weights:      {self.class_weights}\n   Weight init:        {self.weight_initialization}' \
               f'\n   Mix precision:      {mix_precision}'

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
            self.weight_decay,
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
        assert 1 <= epochs <= 2048, f'Number of epochs {epochs} should be [1, 2048]'
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

