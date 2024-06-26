__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from torch import nn
from typing import Tuple, NoReturn, Self
from dl.block.builder.conv1dblockbuilder import Conv1DBlockBuilder
from dl.block.builder.conv2dblockbuilder import Conv2DBlockBuilder
from dl.dlexception import DLException

"""    
    Generic convolutional neural block for 1 and 2 dimensions
    Components:
         Convolution (kernel, Stride, padding)
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
                 kernel_size: int | Tuple[int, int],
                 stride: int | Tuple[int, int],
                 padding: int | Tuple[int, int],
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
        @type batch_norm: bool
        @param max_pooling_kernel: Boolean flag to specify max pooling is needed
        @type max_pooling_kernel: int
        @param activation: Activation function as nn.Module
        @type activation: int
        @param bias: Specify if bias is not null
        @type bias: bool
        @param is_spectral: Specify if we need to apply the spectral norm to the convolutional layer
        @type is_spectral: bool
        """
        ConvBlock.validate_input(
            conv_dimension,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            max_pooling_kernel)

        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_dimension = conv_dimension
        self.is_spectral = is_spectral
        match conv_dimension:
            case 1:
                block_builder = Conv1DBlockBuilder(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    batch_norm,
                    max_pooling_kernel,
                    activation,
                    bias)
            case 2:
                block_builder = Conv2DBlockBuilder(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    batch_norm,
                    max_pooling_kernel,
                    activation,
                    bias)
            case _:
                raise DLException(f'Convolution for dimension {conv_dimension} is not supported')
        block_builder.is_valid()
        self.modules = block_builder()

    def invert(self) -> Self:
        pass

    def __repr__(self) -> str:
        return ' '.join([f'\n{str(module)}' for module in self.modules])

    def get_modules_weights(self) -> Tuple[nn.Module]:
        """
        Get the weights for modules which contains them
        @returns: weight of convolutional neural_blocks
        @rtype: tuple
        """
        return tuple([module for module in self.modules \
                      if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.Conv1d])

    @staticmethod
    def validate_input(
            conv_dimension: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int | Tuple[int, int],
            stride: int | Tuple[int, int],
            padding: int | Tuple[int, int],
            max_pooling_kernel: int = -1) -> NoReturn:
        try:
            assert 0 < conv_dimension < 3, f'Conv neural block conv_dim {conv_dimension} should be {1, 2, 3}'
            assert in_channels > 0, f'Conv neural block in_channels {in_channels} should be >0'
            assert out_channels > 0, f'Conv neural block out_channels {out_channels} should be >0'
            if conv_dimension == 1:
                assert kernel_size > 0, f'Conv neural block kernel_size {kernel_size} should be > 0'
                assert stride >= 0, f'Conv neural block stride {stride} should be >= 0'
                assert padding >= 0, f'Conv neural block padding {padding} should be >= 0'
            else:
                assert kernel_size[0] > 0 and kernel_size[1], f'Conv neural block kernel_size {kernel_size} should be > 0'
                assert stride[0] >= 0 and stride[1] >= 0, f'Conv neural block stride {stride} should be >= 0'
                assert padding[0] >= 0 and padding[1] >= 0, f'Conv neural block padding {padding} should be >= 0'
            assert 0 <= max_pooling_kernel < 5 or max_pooling_kernel == -1, \
                f'Conv neural block max_pooling_kernel size {max_pooling_kernel} should be [0, 4]'
        except AssertionError as e:
            raise DLException(str(e))