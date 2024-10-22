__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from abc import ABC

from dl.block.builder import ConvBlockBuilder
import torch.nn as nn
from typing import Tuple, NoReturn
from dl.block import ConvException


class Conv1DBlockBuilder(ConvBlockBuilder, ABC):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_size: int | Tuple[int],
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
        @param input_size: Size of the input vector
        @type input_size: int
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
                                                 input_size,
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

    def get_conv_out_shape(self) -> int | Tuple[int, int]:
        """
        Compute the output channels from the input channels, stride, padding and kernel size
        @return: output channels if correct, -1 otherwise
        @rtype: int
        """
        num = self.in_channels + 2*self.padding - self.kernel_size
        return int(num/self.stride) + 1 if num % self.stride == 0 else -1

    """ -----------------------  Private methods -------------------------- """

    @staticmethod
    def validate_input(
            in_channels: int,
            out_channels: int,
            input_size: int | Tuple[int, int],
            kernel_size: int | Tuple[int, int],
            stride: int | Tuple[int, int],
            padding: int | Tuple[int, int],
            max_pooling_kernel: int = -1) -> NoReturn:
        try:
            assert in_channels > 0, f'Conv neural block in_channels {in_channels} should be >0'
            assert out_channels > 0, f'Conv neural block out_channels {out_channels} should be >0'
            assert input_size > 0, f'Conv neural block input_size should be {input_size} should be >0'
            assert kernel_size > 0, f'Conv neural block kernel_size {kernel_size} should be > 0'
            assert stride >= 0, f'Conv neural block stride {stride} should be >= 0'
            assert padding >= 0, f'Conv neural block padding {padding} should be >= 0'
            assert 0 <= max_pooling_kernel < 5 or max_pooling_kernel == -1, \
                f'Conv neural block max_pooling_kernel size {max_pooling_kernel} should be [0, 4]'
        except AssertionError as e:
            raise ConvException(str(e))
