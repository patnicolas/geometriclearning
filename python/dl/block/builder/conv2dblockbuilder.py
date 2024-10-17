__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."


from abc import ABC
from dl.block.builder import ConvBlockBuilder
import torch.nn as nn
from typing import Tuple, List, NoReturn
from dl.block import ConvException


class Conv2DBlockBuilder(ConvBlockBuilder, ABC):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_size: Tuple[int, int],
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int],
                 padding: Tuple[int, int],
                 batch_norm: bool,
                 max_pooling_kernel: int,
                 activation: nn.Module,
                 bias: bool) -> None:
        """
        Constructor for the initialization of 2-dimension convolutional neural block
        @param in_channels: Number of input_tensor channels
        @type in_channels: int
        @param out_channels: Number of output channels
        @type out_channels: int
        @param input_size: Height or width of the input
        @type Tuple[int, int]
        @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
        @type kernel_size: Tuple[int, int]
        @param stride: Stride for convolution (st) for 1D, (st, st) for 2D
        @type stride: Tuple[int, int]
        @param padding: Padding for convolution (st) for 1D, (st, st) for 2D
        @type padding: Tuple[int, int]
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
            modules.append(nn.MaxPool2d(kernel_size=self.max_pooling_kernel))
        modules_list: List[nn.Module] = modules
        return tuple(modules_list)

    def compute_out_shape(self) -> int | Tuple[int, int]:
        """
        Compute the output channels from the input channels, stride, padding and kernel size
        @return: output channels if correct, -1 otherwise
        @rtype: Tuple[int, int]
        """
        return self.__compute_out_dim_shape(0), self.__compute_out_dim_shape(1)

    def compute_pooling_shape(self, out_shape: int | Tuple[int, int]) -> int | Tuple[int, int]:
        """
        Compute the dimension for the shape of data output from the pooling layer if defined.
        If undefined the output of the pooling module is the output of the previous
        convolutional layer
        @param out_shape: Output shape from the previous convolutional layer
        @type out_shape: Tuple[int, int]
        @return: Shape of output from the pooling module
        @rtype: Tuple[int, int]
        """
        if self.max_pooling_kernel > 0:
            if out_shape[0] % self.max_pooling_kernel != 0:
                raise ConvException(f'Pooling shape: out {out_shape[0]} should be a multiple of '
                                    f'pooling kernel {self.max_pooling_kernel}')
            if out_shape[1] % self.max_pooling_kernel != 0:
                raise ConvException(f'Pooling shape: out {out_shape[1]} should be a multiple of '
                                    f'pooling kernel {self.max_pooling_kernel}')
            h_shape = int(out_shape[0] / self.max_pooling_kernel)
            w_shape = int(out_shape[1] / self.max_pooling_kernel)
            return h_shape, w_shape
        else:
            return out_shape

    """ -------------------------  Private supporting methods --------------------- """

    def __compute_out_dim_shape(self, dim: int) -> int:
        assert 0 <= dim <= 1, f'Dimension {dim} for computing output channel is out of bounds (0, 1)'

        stride = self.stride[dim]
        padding = self.padding[dim]
        kernel_size = self.kernel_size[dim]
        num = self.input_size[dim] + 2 * padding - kernel_size
        if num % stride != 0:
            raise ConvException(f'Output channel cannot be computed {self.__str__()}')
        return int(num / stride) + 1 if num % stride == 0 else -1

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
            assert input_size[0] > 0 and input_size[1] > 0, \
                f'Conv neural block input_size should be {input_size} should be >0'
            assert kernel_size[0] > 0 and kernel_size[1], f'Conv neural block kernel_size {kernel_size} should be > 0'
            assert stride[0] >= 0 and stride[1] >= 0, f'Conv neural block stride {stride} should be >= 0'
            assert padding[0] >= 0 and padding[1] >= 0, f'Conv neural block padding {padding} should be >= 0'

            assert 0 <= max_pooling_kernel < 5 or max_pooling_kernel == -1, \
                f'Conv neural block max_pooling_kernel size {max_pooling_kernel} should be [0, 4]'
        except AssertionError as e:
            raise ConvException(str(e))
