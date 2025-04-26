__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch import nn
import torch
from typing import Self, AnyStr, Optional, Any, Dict
from dl.block.neural_block import NeuralBlock

"""
Multi layer perceptron (fully connected) block with appropriate activation and drop out.
This class support inversion of the block or linear layer for building decoder from encoder
or generator from discriminator.
The block is composed of a list of nn.Module instances
"""


class MLPBlock(NeuralBlock):
    def __init__(self,
                 block_id: AnyStr,
                 layer_module: nn.Linear,
                 activation_module: Optional[nn.Module] = None,
                 dropout_module: Optional[nn.Dropout] = None) -> None:
        """
        Constructor for the Multi-layer Perceptron
        @param block_id: Identifier for this block
        @type block_id: str
        @param layer_module: Linear layer module y = X.W + b
        @type layer_module: nn.Linear
        @param activation_module: Optional activation module (ReLU, Sigmoid, Softmax,)
        @type activation_module: nn.Module
        @param dropout_module: Optional dropout module for regularization during training
        @type dropout_module: nn.Dropout
        """
        super(MLPBlock, self).__init__(block_id)

        # A MLP block should contain at least a fully connected layer
        modules_list = nn.ModuleList()
        modules_list.append(layer_module)
        # Add activation module if defined
        if activation_module is not None:
            modules_list.append(activation_module)
        # Add drop out module if specified
        if dropout_module is not None:
            modules_list.append(dropout_module)

        self.modules_list = modules_list
        self.block_id = block_id
        self.activation_module = activation_module

    @classmethod
    def build(cls, block_attributes: Dict[AnyStr, Any]) -> Self:
        """
        block_attributes = {
            'block_id': 'MyMLP',
            'in_features': in_features,
            'out_features': out_features,
            'activation': nn.ReLU(),
            'dropout': 0.3
        }
        @param block_attributes: Dictionary of attributes as described above
        @type block_attributes: Dictionary [AnyStr, Any]
        @return: instance of MLPBlock
        @rtype: MLPBlock
        """
        block_id = block_attributes['block_id']
        in_features_attribute = block_attributes['in_features']
        out_features_attributes = block_attributes['out_features']
        activation_attribute = block_attributes['activation']
        dropout_attribute = block_attributes['dropout']
        return cls(block_id,
                   nn.Linear(in_features_attribute, out_features_attributes),
                   activation_attribute,
                   nn.Dropout(dropout_attribute) if dropout_attribute > 0 else None)

    def get_in_features(self) -> int:
        return self.modules_list[0].in_features

    def get_out_features(self) -> int:
        return self.modules_list[0].out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.modules_list:
            x = module(x)
        return x

    @classmethod
    def build_from_params(cls,
                          block_id: AnyStr,
                          in_features: int,
                          out_features: int,
                          activation_module: Optional[nn.Module] = None,
                          dropout_p: float = 0.0):
        """
        Alternative constructor using descriptive parameters
        @param block_id  Optional identifier for the Neural block
        @type block_id str
        @param in_features: Number of input features in the Linear transformation
        @type in_features: int
        @param out_features: Number of output features in the Linear transformation
        @type out_features: int
        @param activation_module: Optional activation function
        @type activation_module: nn.Module
        @param dropout_p: Drop out ratio for regularization in training
        @type dropout_p: float
        """
        dropout_module = nn.Dropout(dropout_p) if dropout_p > 0.0 else None
        return cls(block_id=block_id,
                   layer_module=nn.Linear(in_features, out_features),
                   activation_module=activation_module,
                   dropout_module=dropout_module)

    def transpose(self, activation_update: Optional[nn.Module] = None) -> Self:
        """
        Transpose the layer size (in_feature <-> out_feature) and remove drop_out for decoder
        @return: Inverted feed forward neural network block
        @rtype: MLPBlock
        """
        activation_module = activation_update if activation_update is not None else self.modules[1]
        return MLPBlock(block_id=self.block_id,
                        layer_module=nn.Linear(in_features=self.get_out_features(),
                                               out_features=self.get_in_features(),
                                               bias=False),
                        activation_module=activation_module)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def __repr__(self):
        return '\n'.join([f'{idx}: {str(module)}' for idx, module in enumerate(list(self.modules_list))])
