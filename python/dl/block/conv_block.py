__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch import nn
from typing import Tuple, Any, AnyStr, Optional
from dl.block.neural_block import NeuralBlock
from dl.block.deconv_block import DeConvBlock
from dl.block.builder.conv_output_size import ConvOutputSize
from dl.block.builder import ConvBlockBuilder
import logging
logger = logging.getLogger('dl.block.ConvBlock')

"""    
    Generic convolutional neural block for 1 and 2 dimensions
    Components:
         Convolution (kernel, Stride, padding)
         Batch normalization (Optional)
         Activation
         Max pooling (Optional)

    Formula to compute output_dim of a convolutional block given an in_channels
        output_dim = (in_channels + 2*padding - kernel_size)/stride + 1
    Note: Spectral Normalized convolution is available only for 2D models
"""


class ConvBlock(NeuralBlock):

    def __init__(self, _id: AnyStr, conv_block_builder: ConvBlockBuilder) -> None:
        """
        Constructor for the convolutional neural block
        @param _id: Identifier this convolutional neural block
        @type _id: str
        @param conv_block_builder: Convolutional block (dimension 1 or 2)
        @type conv_block_builder: ConvBlockBuilder
        """
        self.id = _id
        self.conv_block_builder = conv_block_builder

        modules = self.conv_block_builder()
        super(ConvBlock, self).__init__(_id, tuple(modules))

    def invert(self) -> DeConvBlock:
        return DeConvBlock(self.conv_block_builder, activation=None)

    def invert_with_activation(self, activation: Optional[nn.Module] = None) -> DeConvBlock:
        return DeConvBlock(self.conv_block_builder, activation=activation)

    def get_out_channels(self) -> int:
        return self.conv_block_builder.out_channels

    def get_conv_output_size(self) -> ConvOutputSize:
        builder = self.conv_block_builder
        return ConvOutputSize(builder.kernel_size, builder.stride, builder.padding, builder.max_pooling_kernel)

    def __repr__(self) -> str:
        return ' '.join([f'id={self.id}\n{str(module)}' for module in self.modules])

    def get_modules_weights(self) -> Tuple[Any]:
        """
        Get the weights for modules which contains them
        @returns: weight of convolutional neural_blocks
        @rtype: tuple
        """
        return tuple([module.weight.data for module in self.modules \
                      if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.Conv1d])
