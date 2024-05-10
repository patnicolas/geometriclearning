__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from abc import ABC

from dl.block.builder import ConvBlockBuilder
import torch.nn as nn
from typing import Tuple


class DeConv2DBlockBuilder(ConvBlockBuilder, ABC):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int],
                 padding: Tuple[int, int],
                 batch_norm: bool,
                 activation: nn.Module,
                 bias: bool):
        """
        Constructor for the initialization of 1 dimension convolutional neural block
        @param in_channels: Number of input_tensor channels
        @type in_channels: int
        @param out_channels: Number of output channels
        @type out_channels: int
        @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
        @type kernel_size: Union[int, Tuple[int]]
        @param stride: Stride for convolution (st) for 1D, (st, st) for 2D
        @type stride: Union[int, Tuple[int]]
        @param padding: Padding for convolution (st) for 1D, (st, st) for 2D
        @type padding: Union[Int, Tuple[int]]
        @param batch_norm: Boolean flag to specify if a batch normalization is required
        @type batch_norm: int
        @param activation: Activation function as nn.Module
        @type activation: int
        @param bias: Specify if bias is not null
        @type bias: bool
        """
        super(DeConv2DBlockBuilder, self).__init__( in_channels,
                                                    out_channels,
                                                    kernel_size,
                                                    stride,
                                                    padding,
                                                    batch_norm,
                                                    padding,
                                                    -1,
                                                    activation,
                                                    bias)

    def __call__(self) -> Tuple[nn.Module]:
        """
        Generate all torch module for this 1-dimension convolutional neural block
        @param self: Reference to this convolutional neural block builder
        @type self: Conv1DBlockBuilder
        @return: List of torch module
        @rtype: Tuple
        """
        modules = []
        conv_module = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias)
        modules.append(conv_module)
        if self.batch_norm:
            modules.append(nn.BatchNorm2d(self.out_channels))
        if self.activation is not None:
            modules.append(self.activation)
        return tuple(modules)

    def compute_out_channels(self) -> int:
        """
        Compute the output channels from the input channels, stride, padding and kernel size
        @return: output channels if correct, -1 otherwise
        @rtype: int
        """
        stride = self.stride[0]*self.stride[1]
        padding = self.padding[0]*self.padding[1]
        kernel_size = self.kernel_size[0]*self.kernel_size[1]
        return stride*(self.in_channels-1) -2*padding + kernel_size

