__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch.nn as nn
from typing import Tuple, Self, Optional, AnyStr

from dl.block.neural_block import NeuralBlock
from dl.block.conv_block_config import ConvBlockConfig
from dl import ConvException


"""    
    Generic de convolutional neural block for 1 and 2 dimensions
    Components:
         Convolution (kernel, Stride, padding)
         Batch normalization (Optional)
         Activation

    Formula to compute output_dim of a de convolutional block given an in_channels
        output_dim = stride*(in_channels -1) - 2*padding + kernel_size
"""


class DeConv2DBlock(NeuralBlock):
    def __init__(self,
                 block_id: Optional[AnyStr],
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 batch_norm: bool = False,
                 activation: nn.Module = None,
                 bias: bool = False) -> None:
        """
            Alternative constructor for the de convolutional neural block
            @param in_channels: Number of input_tensor channels
            @type in_channels: int or (int, int) for 2D
            @param out_channels: Number of output channels
            @type out_channels: int or (int, int) for 2D
            @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
            @type kernel_size: int or (int, int) for 2D
            @param stride: Stride for convolution (st) for 1D, (st, st) for 2D
            @type stride: int or (int, int) for 2D
            @param padding: Padding for convolution (st) for 1D, (st, st) for 2D
            @type padding: int or (int, int) for 2D
            @param batch_norm: Boolean flag to specify if a batch normalization is required
            @type batch_norm: bool
            @param activation: Activation function as nn.Module
            @type activation: int
            @param bias: Specify if bias is not null
            @type bias: bool
        """
        self.conv_block_config = ConvBlockConfig.de_conv(in_channels,
                                                         out_channels,
                                                         kernel_size,
                                                         stride,
                                                         padding,
                                                         batch_norm,
                                                         activation,
                                                         bias)
        modules = []
        conv_module = nn.ConvTranspose2d(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=bias)
        modules.append(conv_module)

        if batch_norm:
            modules.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            modules.append(activation)
        super(DeConv2DBlock, self).__init__(block_id, tuple(modules))

    @classmethod
    def from_conv(cls, block_id: AnyStr, conv_block_config: ConvBlockConfig, activation: Optional[nn.Module] = None) -> Self:
        """
        Alternate constructor using a pre-configured block and an optional overwriting activation function. If the
        activation function is not specified, the activation function of the convolutional block is used
        @param block_id: Identifier for this de-convolutional block
        @type block_id: str
        @param conv_block_config: Configuration
        @type conv_block_config: ConvBlockConfig
        @param activation: New Activation function
        @type activation: nn.Module
        @return: Instance of 2D De-convolutional block
        @rtype: DeConv2DBlock
        """
        match conv_block_config.get_dimension() :
            case 2:
                activation_module = activation if activation is not None else conv_block_config.activation
                return cls(block_id,
                           conv_block_config.out_channels,
                           conv_block_config.in_channels,
                           conv_block_config.kernel_size,
                           conv_block_config.stride,
                           conv_block_config.padding,
                           conv_block_config.batch_norm,
                           activation_module,
                           conv_block_config.bias)
            case _:
                raise ConvException(f'Cannot create a De convolutional block')

    def invert(self, extra: Optional[nn.Module] = None) -> Self:
        """
        Cannot build an inverted de-convolutional neural block.
        @param extra: Extra module to be added to the inverted neural structure
        @type extra: nn.Module
        @return: ConvException
        """
        raise ConvException('Cannot invert a de-convolutional neural block')

    def __str__(self) -> AnyStr:
        modules_str = self.__repr__()
        config_str = str(self.conv_block_config)
        return f'\nConfiguration {self.block_id}:\n{config_str}\nModules:\n{modules_str}'

    def __repr__(self) -> str:
        return ' '.join([f'\n{idx}: {str(module)}' for idx, module in enumerate(self.modules)])