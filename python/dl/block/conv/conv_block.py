__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch import nn
from typing import Tuple, Any, AnyStr, Optional, Dict

from dl import ConvException
from dl.block.neural_block import NeuralBlock
from dl.block.conv.conv_output_size import ConvOutputSize
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
    def __init__(self,
                 block_id: Optional[AnyStr],
                 modules: Tuple[nn.Module],
                 attributes: Optional[Dict[AnyStr, nn.Module]] = None) -> None:
        """
        Constructor for the Generic Convolutional Neural block
        @param block_id: Identifier for the block
        @type block_id: str
        @param modules: Optional tuple/sequence of convolutional related modules
        @type modules: Tuple[nn.Module]
        @param attributes: Optional dictionary of convolutional attributes used for building
                            De convolutional blocks
        @type attributes: Dict
        """
        self.attributes = attributes
        super(ConvBlock, self).__init__(block_id, tuple(modules))

    def transpose(self, extra: Optional[nn.Module] = None) -> Any:
        raise ConvException('Cannot invert abstract Convolutional block')

    def get_out_channels(self) -> int:
        return self.conv_block_config.out_channels

    def get_conv_output_size(self) -> ConvOutputSize:
        config = self.conv_block_config
        return ConvOutputSize(config.kernel_size, config.stride, config.padding, config.max_pooling_kernel)

    def __str__(self) -> AnyStr:
        modules_str = self.__repr__()
        return f'\n{self.block_id}:\nModules:\n{modules_str}'

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
