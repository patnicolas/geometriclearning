__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch import nn
import torch
from typing import Self, AnyStr, Optional
from dl.block.neural_block import NeuralBlock

"""
Feed Forward (fully connected) Neural Network layer with appropriate activation and drop out.
This class support inversion of the block or linear layer for building decoder from encoder
or generator from discriminator.
The block is composed of a list of nn.Module instances
"""


class FFNNBlock(NeuralBlock):
    def __init__(self,
                 block_id: AnyStr,
                 layer: nn.Linear,
                 activation: nn.Module,
                 drop_out: float = 0.0):
        """
        Default constructor
        @param block_id  Optional identifier for the Neural block
        @type block_id str
        @param layer: Linear module
        @type layer: nn.Linear
        @param activation: Activation function
        @type activation: nn.Module
        @param drop_out: Drop out factor for training purpose
        @type drop_out: float
        """
        self.in_features = layer.in_features
        self.out_features = layer.out_features

        # Starts a build the list of modules
        modules = [layer]
        if activation is not None:
            modules.append(activation)

        # Only if regularization is needed
        if drop_out > 0.0:
            modules.append(nn.Dropout(drop_out))
        super(FFNNBlock, self).__init__(block_id, tuple(modules))
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    @classmethod
    def build(cls,
              block_id: AnyStr,
              in_features: int,
              out_features: int,
              activation: Optional[nn.Module] = None,
              drop_out: float = 0.0):
        """
        Alternative constructor for this block
        @param block_id  Optional identifier for the Neural block
        @type block_id str
        @param in_features: Number of input features or variables
        @type in_features: int
        @param out_features:  Number of output features or variables
        @type out_features: int
        @param activation: activation module
        @type activation: nn.Module
        @param drop_out: Drop out factor
        @type drop_out:
        @return: Instance of a Feed Forward Neural Network block
        @rtype: FFNNBlock
        """
        return cls(block_id, nn.Linear(in_features, out_features, False), activation, drop_out)

    def invert(self, extra: Optional[nn.Module] = None) -> Self:
        """
        Invert the layer size (in_feature <-> out_feature) and remove drop_out for decoder
        @return: Inverted feed forward neural network block
        @rtype: FFNNBlock
        """
        activation_module = self.modules[1]
        return FFNNBlock(self.block_id, nn.Linear(self.out_features, self.in_features, False), activation_module)

    def __repr__(self) -> AnyStr:
        return f'   {super().__repr__()}\n         Num. input: {self.in_features}, Num. output: {self.out_features}'

    def reset_parameters(self):
        self.linear.reset_parameters()
