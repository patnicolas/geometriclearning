__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch import nn
from typing import Tuple, AnyStr, Self

from dl import ConvException
import logging
logger = logging.getLogger('dl.block.ConvBlockConfig')


class ConvBlockConfig(object):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | Tuple[int, int],
                 stride: int | Tuple[int, int],
                 padding: int | Tuple[int, int],
                 batch_norm: bool,
                 max_pooling_kernel: int,
                 activation: nn.Module,
                 bias: bool) -> None:
        """
        Constructor for the configuration/initialization of the convolutional block
        @param in_channels: Number of input_tensor channels
        @type in_channels: int
        @param out_channels: Number of output channels
        @type out_channels: int
        @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
        @type kernel_size: Union[int, Tuple[Int]]
        @param stride: Stride for convolution (st) for 1D, (st, st) for 2D
        @type stride: Union[int, Tuple[Int]]
        @param batch_norm: Boolean flag to specify if a batch normalization is required
        @type batch_norm: int
        @param max_pooling_kernel: Boolean flag to specify max pooling is needed
        @type max_pooling_kernel: int
        @param activation: Activation function as nn.Module
        @type activation: int
        @param bias: Specify if bias is not null
        @type bias: bool
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_norm = batch_norm
        self.max_pooling_kernel = max_pooling_kernel
        self.activation = activation
        self.bias = bias

    @classmethod
    def de_conv(cls,
                in_channels: int,
                out_channels: int,
                kernel_size: int | Tuple[int, int],
                stride: int | Tuple[int, int],
                padding: int | Tuple[int, int],
                batch_norm: bool,
                activation: nn.Module,
                bias: bool) -> Self:
        """
        Constructor for the configuration/initialization of the de-convolutional block
        @param in_channels: Number of input_tensor channels
        @type in_channels: int
        @param out_channels: Number of output channels
        @type out_channels: int
        @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
        @type kernel_size: Union[int, Tuple[Int]]
        @param stride: Stride for convolution (st) for 1D, (st, st) for 2D
        @type stride: Union[int, Tuple[Int]]
        @param batch_norm: Boolean flag to specify if a batch normalization is required
        @type batch_norm: int
        @param activation: Activation function as nn.Module
        @type activation: int
        @param bias: Specify if bias is not null
        @type bias: bool
        """
        return cls(in_channels, out_channels, kernel_size, stride, padding, batch_norm, -1, activation, bias)

    def __str__(self) -> AnyStr:
        return (f'\nIn channels: {self.in_channels}\nOut channels: {self.out_channels}\nKernel size: {self.kernel_size}\''
                f'\nStride: {self.stride}\nPadding: {self.padding}\nBatch norm: {self.batch_norm}'
                f'\nMax Pooling kernel: {self.max_pooling_kernel}\nActivation: {self.activation}')

    def get_dimension(self) -> int:
        if isinstance(self.kernel_size, Tuple):
            return len(self.kernel_size)
        elif isinstance(self.kernel_size, int):
            return 1
        else:
            raise ConvException(f'Dimension ')