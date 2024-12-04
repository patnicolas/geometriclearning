__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch import nn
from typing import Tuple, Any, AnyStr, Optional

from dl import ConvException
from dl.block.neural_block import NeuralBlock
from dl.block.conv_block_config import ConvBlockConfig
from dl.block.builder.conv_output_size import ConvOutputSize
import logging
logger = logging.getLogger('dl.block.ConvBlock')



"""    
    Generic convolutional neural block for 1 and 2 dimensions
    Components:
         Convolution (kernel, Stride, padding)
         Batch normalization (Optional)
         Activation

    Formula to compute output_dim of a convolutional block given an in_channels
        output_dim = (in_channels + 2*padding - kernel_size)/stride + 1
    Note: Spectral Normalized convolution is available only for 2D models
"""


class ConvBlock(NeuralBlock):
    def __init__(self, block_id: Optional[AnyStr], conv_block_config: ConvBlockConfig, modules: Tuple[nn.Module]) -> None:
        """
        Constructor for the Generic Convolutional Neural block
        @param block_id: Identifier for the block
        @type block_id: str
        @param conv_block_config: Configuration for the convolutional block
        @type conv_block_config: ConvBlockConfig
        @param modules: PyTorch modules components for this block
        @type modules: Tuple (modules)
        """
        self.conv_block_config = conv_block_config
        super(ConvBlock, self).__init__(block_id, modules)

    def transpose(self, extra: Optional[nn.Module] = None) -> Any:
        raise ConvException('Cannot invert abstract Convolutional block')

    def get_out_channels(self) -> int:
        return self.conv_block_config.out_channels

    def get_conv_output_size(self) -> ConvOutputSize:
        config = self.conv_block_config
        return ConvOutputSize(config.kernel_size, config.stride, config.padding, config.max_pooling_kernel)

    def __str__(self) -> AnyStr:
        modules_str = self.__repr__()
        config_str = str(self.conv_block_config)
        return f'\nConfiguration {self.block_id}:\n{config_str}\nModules:\n{modules_str}'

    def __repr__(self) -> AnyStr:
        return '\n'.join([f'{idx}: {str(module)}' for idx, module in enumerate(self.modules)])

    def get_modules_weights(self) -> Tuple[Any]:
        """
        Get the weights for modules which contains them
        @returns: weight of convolutional neural_blocks
        @rtype: tuple
        """
        return tuple([module.weight.data for module in self.modules \
                      if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.Conv1d])
