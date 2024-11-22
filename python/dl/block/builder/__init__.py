_author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import abc
import torch.nn as nn
from typing import Tuple
from dl import ConvException


class ConvBlockBuilder(object):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_size: int | Tuple[int, int],
                 kernel_size: int | Tuple[int, int],
                 stride: int | Tuple[int, int],
                 padding: int | Tuple[int, int],
                 batch_norm: bool,
                 max_pooling_kernel: int,
                 activation: nn.Module,
                 bias: bool):
        """
        Constructor for the initialization of
        @param in_channels: Number of input_tensor channels
        @type in_channels: int
        @param out_channels: Number of output channels
        @type out_channels: int
        @param input_size: Size (weight or height) if the input
        @type input_size: Union[int, Tuple[Int]]
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
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_norm = batch_norm
        self.max_pooling_kernel = max_pooling_kernel
        self.activation = activation
        self.bias = bias

    @abc.abstractmethod
    def __call__(self) -> Tuple[nn.Module]:
        raise ConvException('Cannot extract module from abstract class ConvInitBlockBuilder')

    @abc.abstractmethod
    def get_conv_layer_out_shape(self) -> int | Tuple[int, int]:
        raise ConvException('Computation of shape of convolution layer output is undefined')


def extract_conv_dimensions(
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int],
        padding: int | Tuple[int, int]) -> Tuple[int]:
    if isinstance(kernel_size, int):
        return kernel_size, stride, padding
    else:
        return sum(kernel_size), sum(stride), sum(stride)
