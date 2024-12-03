__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from dl.block.conv_block import ConvBlock, ConvBlockConfig
from dl.block.deconv_2d_block import DeConv2DBlock
from typing import AnyStr, Tuple, Optional, List, Self
import torch.nn as nn


class Conv2DBlock(ConvBlock):
    def __init__(self,
                 block_id: AnyStr,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 batch_norm: bool = False,
                 max_pooling_kernel: int = 1,
                 activation: nn.Module = None,
                 bias: bool = False) -> None:
        """
        Constructor for a 2-dimension convolutional block
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
        conv_block_config = ConvBlockConfig(in_channels,
                                            out_channels,
                                            kernel_size,
                                            stride,
                                            padding,
                                            batch_norm,
                                            max_pooling_kernel,
                                            activation,
                                            bias)
        modules = []

        # First define the 2D convolution
        conv_module = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        modules.append(conv_module)

        # Add the batch normalization
        if batch_norm:
            batch_module: nn.Module = nn.BatchNorm2d(out_channels)
            modules.append(batch_module)

        # Activation to be added if needed
        if activation is not None:
            activation_module: nn.Module = activation
            modules.append(activation_module)

        # Added max pooling module if defined
        if max_pooling_kernel > 0:
            max_pool_module: nn.Module = nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=1, padding=0)
            modules.append(max_pool_module)

        # modules_list: List[nn.Module] = modules
        super(Conv2DBlock, self).__init__(block_id, conv_block_config, tuple(modules))

    def invert(self, extra: Optional[nn.Module] = None) -> DeConv2DBlock:
        """
        Build a de-convolutional neural block from an existing convolutional block
        @param extra: Extra module to be added to the inverted neural structure
        @type extra: nn.Module
        @return: Instance of 2D de-convolutional block
        @rtype: DeConv2DBlock
        """
        return DeConv2DBlock.from_conv(block_id=f'de_{self.block_id}',
                                       conv_block_config=self.conv_block_config,
                                       activation=extra)



