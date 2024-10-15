__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from torch import nn
from typing import Tuple, NoReturn, Self
from dl.block.builder.conv1dblockbuilder import Conv1DBlockBuilder
from dl.block.builder.conv2dblockbuilder import Conv2DBlockBuilder
from dl.block.builder import ConvBlockBuilder
from dl.dlexception import DLException

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
        self.conv_block_builder .is_valid()
        # Invoke __call__
        self.modules = self.conv_block_builder()

    def invert(self) -> Self:
        pass

    def __repr__(self) -> str:
        return ' '.join([f'\n{str(module)}' for module in self.modules])

    def get_modules_weights(self) -> Tuple[nn.Module]:
        """
        Get the weights for modules which contains them
        @returns: weight of convolutional neural_blocks
        @rtype: tuple
        """
        return tuple([module for module in self.modules \
                      if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.Conv1d])
