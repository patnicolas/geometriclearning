__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch.nn as nn
from typing import Tuple, Self, Optional

from dl.block.builder import ConvBlockBuilder
from dl.block.builder.deconv1d_block_builder import DeConv1DBlockBuilder
from dl.block.builder.conv2d_block_builder import Conv2DBlockBuilder
from dl.block.builder.conv1d_block_builder import Conv1DBlockBuilder
from dl.block.builder.deconv2d_block_builder import DeConv2DBlockBuilder
from dl import ConvException


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

    def __init__(self, conv_block_builder: ConvBlockBuilder, activation: Optional[nn.Module] = None) -> None:
        super(DeConvBlock, self).__init__()

        if isinstance(conv_block_builder, Conv2DBlockBuilder):
            activation_module = activation if activation is not None else conv_block_builder.activation
            de_conv_block_builder = DeConv2DBlockBuilder.build(conv_block_builder, activation_module)
        else:
            raise ConvException(f'Cannot create a De convolutional block from type {type(conv_block_builder)}')

        self.in_channels = de_conv_block_builder.in_channels
        self.out_channels = de_conv_block_builder.out_channels
        self.modules = de_conv_block_builder()

    @classmethod
    def build(cls,
              conv_dimension: int,
              in_channels: int,
              out_channels: int,
              input_size: int | Tuple[int, int],
              kernel_size: int | Tuple[int, int],
              stride: int | Tuple[int, int],
              padding: int | Tuple[int, int],
              batch_norm: bool,
              activation: nn.Module,
              bias: bool) -> Self:
        """
            Alternative constructor for the de convolutional neural block
            @param conv_dimension: Dimension of the de convolution (1 or 2)
            @type conv_dimension: int
            @param in_channels: Number of input_tensor channels
            @type in_channels: int or (int, int) for 2D
            @param out_channels: Number of output channels
            @type out_channels: int or (int, int) for 2D
            @param input_size: Size of input
            @type input_size: int or (int, int) for 2D
            @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
            @type kernel_size: int or (int, int) for 2D
            @param stride: Stride for convolution (st) for 1D, (st, st) for 2D
            @type stride: int or (int, int) for 2D
            @param padding: Padding for convolution (st) for 1D, (st, st) for 2D
            @type padding: int or (int, int) for 2D
            @param batch_norm: Boolean flag to specify if a batch normalization is required
            @type batch_norm: bool
            @param activation: Activation function as nn.Module
            @type activation: int
            @param bias: Specify if bias is not null
            @type bias: bool
        """

        match conv_dimension:
            case 1:
                block_builder = DeConv1DBlockBuilder(
                    in_channels,
                    out_channels,
                    input_size,
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
                    input_size,
                    kernel_size,
                    stride,
                    padding,
                    batch_norm,
                    activation,
                    bias)
            case _:
                raise ConvException(f'De Convolution for dimension {conv_dimension} is not supported')
        return cls(block_builder)

    def invert(self) -> Self:
        raise ConvException('Cannot invert a de-convolutional neural block')

    def __repr__(self) -> str:
        return ' '.join([f'\n{str(module)}' for module in self.modules])
