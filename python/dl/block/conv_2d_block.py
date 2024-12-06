__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from dl.block.conv_block import ConvBlock, ConvBlockConfig
from dl.block.deconv_2d_block import DeConv2DBlock
from typing import AnyStr, Tuple, Optional, Self
import torch.nn as nn
from dl import Conv2DataType


class Conv2DBlock(ConvBlock):
    def __init__(self, block_id: AnyStr, conv_block_config: ConvBlockConfig) -> None:
        """
        Constructor for a 2-dimension convolutional block
        @param block_id: Identifier for the 2D convolutional block
        @type block_id: str
        @param conv_block_config: Configuration for this convolutional neural block
        @type conv_block_config: ConvBlockConfig
        """
        modules = []

        # First define the 2D convolution
        conv_module = nn.Conv2d(in_channels=conv_block_config.in_channels,
                                out_channels=conv_block_config.out_channels,
                                kernel_size=conv_block_config.kernel_size,
                                stride=conv_block_config.stride,
                                padding=conv_block_config.padding,
                                bias=conv_block_config.bias)
        modules.append(conv_module)

        # Add the batch normalization
        if conv_block_config.batch_norm:
            batch_module: nn.Module = nn.BatchNorm2d(conv_block_config.out_channels)
            modules.append(batch_module)

        # Activation to be added if needed
        if conv_block_config.activation is not None:
            activation_module: nn.Module = conv_block_config.activation
            modules.append(activation_module)

        # Added max pooling module if defined
        if conv_block_config.max_pooling_kernel > 0:
            max_pool_module: nn.Module = nn.MaxPool2d(kernel_size=conv_block_config.max_pooling_kernel,
                                                      stride=1,
                                                      padding=0)
            modules.append(max_pool_module)

        if conv_block_config.drop_out > 0.0:
            modules.append(nn.Dropout2d(conv_block_config.drop_out))

        # modules_list: List[nn.Module] = modules
        super(Conv2DBlock, self).__init__(block_id, conv_block_config, tuple(modules))

    @classmethod
    def build(cls,
              block_id: AnyStr,
              in_channels: int,
              out_channels: int,
              kernel_size: Conv2DataType,
              stride: Conv2DataType = (1, 1),
              padding: Conv2DataType = (0, 0),
              batch_norm: bool = False,
              max_pooling_kernel: int = 1,
              activation: nn.Module = None,
              bias: bool = False,
              drop_out: float = 0.0) -> Self:
        """
        Alternative constructor for a 2-dimension convolutional block
        @param block_id: Identifier for the block id
        @type block_id: str
        @param in_channels: Number of input_tensor channels
        @type in_channels: int
        @param out_channels: Number of output channels
        @type out_channels: int
        @param kernel_size: Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
        @type kernel_size: Tuple[int, int]
        @param stride: Stride for convolution (st) for 1D, (st, st) for 2D
        @type stride: Tuple[int, int]
        @param padding: Padding for convolution (st) for 1D, (st, st) for 2D
        @type stride: Tuple[int, int]
        @param batch_norm: Boolean flag to specify if a batch normalization is required
        @type batch_norm: int
        @param max_pooling_kernel: Boolean flag to specify max pooling is needed
        @type max_pooling_kernel: int
        @param activation: Activation function as nn.Module
        @type activation: int
        @param bias: Specify if bias is not null
        @type bias: bool
        @param drop_out: Regularization term applied if > 0
        @type drop_out: float
        """
        conv_block_config = ConvBlockConfig(in_channels,
                                            out_channels,
                                            kernel_size,
                                            stride,
                                            padding,
                                            batch_norm,
                                            max_pooling_kernel,
                                            activation,
                                            bias,
                                            drop_out)
        return cls(block_id, conv_block_config)

    def transpose(self, extra: Optional[nn.Module] = None) -> DeConv2DBlock:
        """
        Build a de-convolutional neural block from an existing convolutional block
        @param extra: Extra module to be added to the inverted neural structure
        @type extra: nn.Module
        @return: Instance of 2D de-convolutional block
        @rtype: DeConv2DBlock
        """
        self.conv_block_config.transpose()
        return DeConv2DBlock(block_id=f'de_{self.block_id}',
                             conv_block_config=self.conv_block_config,
                             activation=extra)



