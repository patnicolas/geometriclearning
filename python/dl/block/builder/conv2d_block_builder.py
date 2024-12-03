__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."


from abc import ABC
from dl.block.builder import ConvBlockBuilder
import torch.nn as nn
from typing import Tuple, List, NoReturn
from dl import ConvException


class Conv2DBlockBuilder(ConvBlockBuilder, ABC):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_size: int | Tuple[int, int],
                 kernel_size: int | Tuple[int, int],
                 stride: int | Tuple[int, int] = (1, 1),
                 padding: int | Tuple[int, int] = (0, 0),
                 batch_norm: bool = False,
                 max_pooling_kernel: int = 1,
                 activation: nn.Module = None,
                 bias: bool = False) -> None:
        """
        Constructor for the initialization of 2-dimension convolutional neural block
        @param in_channels: Number of input_tensor channels
        @type in_channels: int
        @param out_channels: Number of output channels
        @type out_channels: int
        @param input_size: Height or width of the input
        @type input_size: int or (int, int) for 2D
        @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
        @type kernel_size: int for 1D or Tuple[int, int] for 2D data
        @param stride: Stride for convolution (st) for 1D, (st, st) for 2D
        @type stride: int for 1D or Tuple[int, int] for 2D data
        @param padding: Padding for convolution (st) for 1D, (st, st) for 2D
        @type padding: int for 1D or Tuple[int, int] for 2D data
        @param batch_norm: Boolean flag to specify if a batch normalization is required
        @type batch_norm: int
        @param max_pooling_kernel: Boolean flag to specify max pooling is needed
        @type max_pooling_kernel: int
        @param activation: Activation function as nn.Module
        @type activation: int
        @param bias: Specify if bias is not null
        @type bias: bool
        """
        Conv2DBlockBuilder.__validate_input(
            in_channels,
            out_channels,
            input_size,
            kernel_size,
            stride,
            padding,
            max_pooling_kernel)
        super(Conv2DBlockBuilder, self).__init__(in_channels,
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
        Generate all torch module for this 2-dimension convolutional neural block
        @param self: Reference to this convolutional neural block builder
        @type self: Conv1DBlockBuilder
        @return: List of torch module
        @rtype: Tuple
        """
        modules = []
        # First define the 2D convolution
        conv_module = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias)
        modules.append(conv_module)

        # Add the batch normalization
        if self.batch_norm:
            modules.append(nn.BatchNorm2d(self.out_channels))
        # Activation to be added if needed
        if self.activation is not None:
            modules.append(self.activation)

        # Added max pooling module
        if self.max_pooling_kernel > 0:
            modules.append(nn.MaxPool2d(kernel_size=self.max_pooling_kernel, stride=1, padding=0))
        modules_list: List[nn.Module] = modules
        return tuple(modules_list)

    def get_conv_output_size(self, input_size: int | Tuple[int, int]) -> int | Tuple[int, int]:
        from dl.block.builder.conv_output_size import ConvOutputSize

        conv_output_size = ConvOutputSize(self.kernel_size, self.stride, self.padding, self.max_pooling_kernel)
        return conv_output_size(input_size)

    """ -------------------------  Private supporting methods --------------------- """
    @staticmethod
    def __validate_input(
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
            if type(input_size) is int:
                assert input_size > 0, \
                    f'Conv neural block input_size should be {input_size} should be >0'
                assert kernel_size > 0, f'Conv neural block kernel_size {kernel_size} should be > 0'
                assert stride >= 0, f'Conv neural block stride {stride} should be >= 0'
                assert padding >= 0, f'Conv neural block padding {padding} should be >= 0'
            else:
                assert input_size[0] > 0 and input_size[1] > 0, \
                    f'Conv neural block input_size should be {input_size} should be >0'
                assert kernel_size[0] > 0 and kernel_size[1], f'Conv neural block kernel_size {kernel_size} should be > 0'
                assert stride[0] >= 0 and stride[1] >= 0, f'Conv neural block stride {stride} should be >= 0'
                assert padding[0] >= 0 and padding[1] >= 0, f'Conv neural block padding {padding} should be >= 0'

            assert 0 <= max_pooling_kernel < 5 or max_pooling_kernel == -1, \
                f'Conv neural block max_pooling_kernel size {max_pooling_kernel} should be [0, 4]'
        except AssertionError as e:
            raise ConvException(str(e))
