__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch.nn as nn
from abc import abstractmethod
from typing import Tuple
from dl.dlexception import DLException


class ConvBlockBuilder(object):
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
        Constructor for the initialization of
        @param in_channels: Number of input_tensor channels
        @type in_channels: int
        @param out_channels: Number of output channels
        @type out_channels: int
        @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
        @type kernel_size: Union[int, Tuple[Int]]
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_norm = batch_norm
        self.max_pooling_kernel = max_pooling_kernel
        self.activation = activation
        self.bias = bias

    @abstractmethod
    def __call__(self) -> Tuple[nn.Module]:
        raise DLException('Cannot extract module from abstract class ConvInitBlock')
