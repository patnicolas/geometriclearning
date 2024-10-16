__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from torch import nn
from typing import Tuple, Self, Any
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


class ConvBlock(nn.Module):

    def __init__(self, conv_block_builder: ConvBlockBuilder) -> None:
        """
        Constructor for the convolutional neural block
        @param conv_block_builder: Convolutional block (dimension 1 or 2)
        @type conv_block_builder: ConvBlockBuilder
        """
        super(ConvBlock, self).__init__()
        self.conv_block_builder = conv_block_builder
        # Invoke __call__
        self.modules = self.conv_block_builder()

    def compute_out_shapes(self) -> int | Tuple[int, int]:
        out_shape = self.conv_block_builder.compute_out_shape()
        logging.info(f'Conv output shape: {str(out_shape)}')
        out_pooling_shape = self.conv_block_builder.compute_pooling_shape(out_shape)
        logging.info(f'Max pooling output shape: {str(out_pooling_shape)}')
        return out_pooling_shape

    def invert(self) -> Self:
        pass

    def __repr__(self) -> str:
        return ' '.join([f'\n{str(module)}' for module in self.modules])

    def get_modules_weights(self) -> Tuple[Any]:
        """
        Get the weights for modules which contains them
        @returns: weight of convolutional neural_blocks
        @rtype: tuple
        """
        return tuple([module.weight.data for module in self.modules \
                      if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.Conv1d])
