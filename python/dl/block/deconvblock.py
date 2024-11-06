__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch.nn as nn
from typing import Tuple, Self
from dl.block.builder.deconv1dblockbuilder import DeConv1DBlockBuilder
from dl.block.builder.deconv2dblockbuilder import DeConv2DBlockBuilder
from dl.block.convblock import ConvBlock
from dl.dlexception import DLException


"""    
    Generic de convolutional neural block for 1 and 2 dimensions
    Components:
         Convolution (kernel, Stride, padding)
         Batch normalization (Optional)
         Activation

    Formula to compute output_dim of a de convolutional block given an in_channels
        output_dim = stride*(in_channels -1) - 2*padding + kernel_size
"""


class DeConvBlock(nn.Module):
    def __init__(self,
                 conv_dimension: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | Tuple[int, int],
                 stride: int | Tuple[int, int],
                 padding: int | Tuple[int, int],
                 batch_norm: bool,
                 activation: nn.Module,
                 bias: bool):
        """
            Constructor for the de convolutional neural block
            @param conv_dimension: Dimension of the de convolution (1 or 2)
            @type conv_dimension: int
            @param in_channels: Number of input_tensor channels
            @type in_channels: int
            @param out_channels: Number of output channels
            @type out_channels: int
            @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
            @type kernel_size: int
            @param stride: Stride for convolution (st) for 1D, (st, st) for 2D
            @type stride: int
            @param batch_norm: Boolean flag to specify if a batch normalization is required
            @type batch_norm: bool
            @param activation: Activation function as nn.Module
            @type activation: int
            @param bias: Specify if bias is not null
            @type bias: bool
            """
        ConvBlock.validate_input(conv_dimension, in_channels, out_channels, kernel_size, stride, padding)

        super(DeConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_dimension = conv_dimension

        match conv_dimension:
            case 1:
                block_builder = DeConv1DBlockBuilder(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    batch_norm,
                    activation,
                    bias)
            case 2:
                block_builder = DeConv2DBlockBuilder(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    batch_norm,
                    activation,
                    bias)
            case _:
                raise DLException(f'De Convolution for dimension {conv_dimension} is not supported')
        block_builder.is_valid()
        self.modules = block_builder()

    def invert(self) -> Self:
        pass

    def __repr__(self) -> str:
        return ' '.join([f'\n{str(module)}' for module in self.modules])
