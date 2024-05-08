__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."


from abc import ABC
from dl.block import ConvBlockBuilder
import torch.nn as nn
from typing import Tuple


class Conv2DBlockBuilder(ConvBlockBuilder, ABC):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 batch_norm: bool,
                 max_pooling_kernel: int,
                 activation: nn.Module,
                 bias: bool):
        """
        Constructor for the initialization of 2-dimension convolutional neural block
        @param in_channels: Number of input_tensor channels
        @type in_channels: int
        @param out_channels: Number of output channels
        @type out_channels: int
        @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
        @type kernel_size: int
        @param stride: Stride for convolution (st) for 1D, (st, st) for 2D
        @type stride: int
        @param batch_norm: Boolean flag to specify if a batch normalization is required
        @type batch_norm: int
        @param max_pooling_kernel: Boolean flag to specify max pooling is needed
        @type max_pooling_kernel: int
        @param activation: Activation function as nn.Module
        @type activation: int
        @param bias: Specify if bias is not null
        @type bias: bool
        """
        super(Conv2DBlockBuilder, self).__init__(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride,
                                                 padding,
                                                 batch_norm,
                                                 max_pooling_kernel,
                                                 activation,
                                                 bias)

        def get_conv_modules(self) -> Tuple[nn.Module]:
            """
            Generate all torch module for this 2-dimension convolutional neural block
            @param self: Reference to this convolutional neural block builder
            @type self: Conv1DBlockBuilder
            @return: List of torch module
            @rtype: Tuple
            """
            modules = []
            # First define the 2D convolution
            conv_module = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias)
            modules.append(conv_module)
            # Add the batch normalization
            if batch_norm:
                modules.append(nn.BatchNorm2d(out_channels))
            # Activation to be added if needed
            if activation is not None:
                modules.append(activation)
            # Added max pooling module
            if max_pooling_kernel > 0:
                modules.append(nn.MaxPool2d(max_pooling_kernel))
            return tuple(modules)
