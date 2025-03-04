__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch import nn
import torch
from typing import Self, AnyStr, Optional, Tuple
from dl.block.neural_block import NeuralBlock

"""
Multi layer perceptron (fully connected) block with appropriate activation and drop out.
This class support inversion of the block or linear layer for building decoder from encoder
or generator from discriminator.
The block is composed of a list of nn.Module instances
"""


class MLPBlock(NeuralBlock):
    def __init__(self,
                 block_id: Optional[AnyStr],
                 modules: Tuple[nn.Module]):
        """
        Constructor for the Feed Forward Neural Network
        @param block_id: Identifier for this block
        @type block_id: str
        @param modules: Tuple of torch modules contained in this block
        @type modules: Tuple
        """
        self.activation = [module for module in modules
                           if module.__class__.__name__
                           in NeuralBlock.supported_activations]
        super(MLPBlock, self).__init__(block_id, modules)

        # We get the number of input and output features from the first module of type Linear
        self.in_features = modules[0].in_features
        self.out_features = modules[0].out_features

    @classmethod
    def build(cls,
              block_id: AnyStr,
              layer: nn.Linear,
              activation: Optional[nn.Module] = None,
              drop_out: float = 0.0):
        """
        Alternative constructor
        @param block_id  Optional identifier for the Neural block
        @type block_id str
        @param layer: Linear module
        @type layer: nn.Linear
        @param activation: Activation function
        @type activation: nn.Module
        @param drop_out: Drop out factor for training purpose
        @type drop_out: float
        """
        # Starts a build the list of modules
        modules = [layer]
        if activation is not None:
            modules.append(activation)

        # Only if regularization is needed
        if drop_out > 0.0:
            modules.append(nn.Dropout(drop_out))
        return cls(block_id, tuple(modules))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def transpose(self, activation_update: Optional[nn.Module] = None) -> Self:
        """
        Transpose the layer size (in_feature <-> out_feature) and remove drop_out for decoder
        @return: Inverted feed forward neural network block
        @rtype: MLPBlock
        """
        activation_module = activation_update if activation_update is not None else self.modules[1]
        return MLPBlock.build(block_id=self.block_id,
                              layer=nn.Linear(in_features=self.modules[0].out_features,
                                               out_features=self.modules[0].in_features,
                                               bias=False),
                              activation=activation_module)

    def __repr__(self) -> AnyStr:
        return f'   {super().__repr__()}\n         Num. input: {self.in_features}, Num. output: {self.out_features}'

    def reset_parameters(self):
        self.linear.reset_parameters()
