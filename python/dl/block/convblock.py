__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from torch import nn
from typing import Tuple, List

"""    
    Generic convolutional neural block for 1 and 2 dimensions
    Components:
         Convolution
         Batch normalization (Optional)
         Activation
         Max pooling (Optional)

    Formula to compute output_dim of a convolutional block given an in_channels
        output_dim = (in_channels + 2*padding - kernel_size)/stride + 1
    Note: Spectral Normalized convolution is available only for 2D models
"""


class ConvBlock(nn.Module):

    def __init__(self,
                 conv_dimension: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 batch_norm: bool,
                 max_pooling_kernel: int,
                 activation: nn.Module,
                 bias: bool,
                 is_spectral: bool = False):
        """
        Constructor for the convolutional neural block
        @param conv_dimension: Dimension of the convolution (1 or 2)
        @type conv_dimension: int
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
        @param is_spectral: Specify if we need to apply the spectral norm to the convolutional layer
        @type is_spectral: bool
        """
        super(ConvBlock, self).__init__()

        assert 0 < conv_dimension < 4, f'Conv neural block conv_dim {conv_dimension} should be {1, 2, 3}'
        assert in_channels > 0, f'Conv neural block in_channels {in_channels} should be >0'
        assert out_channels > 0, f'Conv neural block out_channels {out_channels} should be >0'
        assert kernel_size > 0, f'Conv neural block kernel_size {kernel_size} should be >0'
        assert stride >= 0, f'Conv neural block stride {stride} should be >= 0'
        assert 0 <= max_pooling_kernel < 5, f'Conv neural block max_pooling_kernel size {max_pooling_kernel} should be [0, 4]'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_dimension = conv_dimension
        self.is_spectral = is_spectral
        self.modules = self.init1d(kernel_size,
                                   stride,
                                   padding,
                                   batch_norm,
                                   max_pooling_kernel,
                                   activation,
                                   bias) if conv_dimension == 1 \
            else self.init2d(kernel_size,
                             stride,
                             padding,
                             batch_norm,
                             max_pooling_kernel,
                             activation,
                             bias)

    def __repr__(self) -> str:
        return ' '.join([f'\n{str(module)}' for module in self.modules])

    def get_modules(self):
        return self.modules

    def get_modules_weights(self) -> Tuple:
        """
        Get the weights for modules which contains them
        @returns: weight of convolutional neural_blocks
        @rtype: tuple
        """
        return tuple([module for module in self.modules \
                      if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.Conv1d])

    def init1d(self,
               kernel_size: int,
               stride: int,
               padding: int,
               batch_norm: bool,
               max_pooling_kernel: int,
               activation: nn.Module,
               bias: bool) -> List[nn.Module]:
        """
            Instantiation of 1D Convolution block
            :type padding: int
            :param kernel_size Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
            :param stride Stride for convolution (st) for 1D, (st, st) for 2D
            :param batch_norm Boolean flag to specify if a batch normalization is required
            :param max_pooling_kernel Boolean flag to specify max pooling is neede
            :param activation Activation function as nn.Module
            :param bias Specify if bias is not null
            :return: List of PyTorch modules related to Convolution
        """
        modules = []
        conv_module = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)

        # If this is a spectral convolution
        if self.is_spectral:
            conv_module = nn.utils.spectral_norm(conv_module)
        modules.append(conv_module)
        if batch_norm:
            modules.append(nn.BatchNorm1d(self.out_channels))
        if activation is not None:
            modules.append(activation)
        if max_pooling_kernel > 0:
            modules.append(nn.MaxPool1d(max_pooling_kernel))
        return modules

    def init2d(self,
               kernel_size: int,
               stride: int,
               padding: int,
               batch_norm: bool,
               max_pooling_kernel: int,
               activation: nn.Module,
               bias: bool) -> Tuple[nn.Module]:
        """
            Instantiation of 2D Convolution block
            :param kernel_size Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
            :param stride Stride for convolution (st) for 1D, (st, st) for 2D
            :param batch_norm Boolean flag to specify if a batch normalization is required
            :param max_pooling_kernel Boolean flag to specify max pooling is neede
            :param activation Activation function as nn.Module
            :param bias Specify if bias is not null
            :return: List of PyTorch modules related to Convolution
        """
        modules = []
        conv_module = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)

        # If this is a spectral Neural block
        if self.is_spectral:
            conv_module = nn.utils.spectral_norm(conv_module)
        modules.append(conv_module)
        if batch_norm:
            modules.append(nn.BatchNorm2d(self.out_channels))
        if activation is not None:
            modules.append(activation)
        if max_pooling_kernel > 0:
            modules.append(nn.MaxPool2d(max_pooling_kernel))
        return tuple(modules)

    # ---------------------------  Supporting methods -----------------------------

    @staticmethod
    def __init1d(in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 batch_norm: bool,
                 max_pooling_kernel: int,
                 activation: nn.Module,
                 bias: bool) -> Tuple[nn.Module]:
        modules = []
        conv_module = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        modules.append(conv_module)
        if batch_norm:
            modules.append(nn.BatchNorm1d(out_channels))
        if activation is not None:
            modules.append(activation)
        if max_pooling_kernel > 0:
            modules.append(nn.MaxPool1d(max_pooling_kernel))
        return tuple(modules)

    @staticmethod
    def __init2d(in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 batch_norm: bool,
                 max_pooling_kernel: int,
                 activation: nn.Module,
                 bias: bool) -> Tuple[nn.Module]:
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


