__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from abc import ABC

from dl.block import ConvBlockBuilder
import torch.nn as nn
from typing import Tuple, overload


class Conv1DBlockBuilder(ConvBlockBuilder, ABC):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | Tuple[int],
                 stride: int | Tuple[int],
                 padding: int | Tuple[int],
                 batch_norm: bool,
                 max_pooling_kernel: int,
                 activation: nn.Module,
                 bias: bool):
        """
        Constructor for the initialization of 1 dimension convolutional neural block
        @param in_channels: Number of input_tensor channels
        @type in_channels: int
        @param out_channels: Number of output channels
        @type out_channels: int
        @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
        @type kernel_size: Union[Int, Tuple[int]]
        @param stride: Stride for convolution (st) for 1D, (st, st) for 2D
        @type stride: Union[Int, Tuple[int]]
        @param padding: Padding for convolution (st) for 1D, (st, st) for 2D
        @type padding: Union[Int, Tuple[int]]
        @param batch_norm: Boolean flag to specify if a batch normalization is required
        @type batch_norm: int
        @param max_pooling_kernel: Boolean flag to specify max pooling is needed
        @type max_pooling_kernel: int
        @param activation: Activation function as nn.Module
        @type activation: int
        @param bias: Specify if bias is not null
        @type bias: bool
        """
        super(Conv1DBlockBuilder, self).__init__(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride,
                                                 padding,
                                                 batch_norm,
                                                 max_pooling_kernel,
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
        conv_module = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias)
        modules.append(conv_module)
        if self.batch_norm:
            modules.append(nn.BatchNorm1d(self.out_channels))
        if self.activation is not None:
            modules.append(self.activation)
        if self.max_pooling_kernel > 0:
            modules.append(nn.MaxPool1d(self.max_pooling_kernel))
        return tuple(modules)
