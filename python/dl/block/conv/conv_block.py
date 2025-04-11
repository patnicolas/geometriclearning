__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch import nn
from typing import Tuple, Any, AnyStr, Optional, Dict, List

from dl import ConvException
from dl.block.neural_block import NeuralBlock
from dl.block.conv.conv_output_size import ConvOutputSize
import logging
logger = logging.getLogger('dl.block.ConvBlock')


"""
Generic convolutional neural block for 1 and 2 dimensions

Formula to compute output_dim of a convolutional block given an in_channels
        output_dim = (in_channels + 2*padding - kernel_size)/stride + 1
Note: Spectral Normalized convolution is available only for 2D models
"""


class ConvBlock(NeuralBlock):
    def __init__(self, block_id: Optional[AnyStr]) -> None:
        """
        Constructor for the Generic Convolutional Neural block
        @param block_id: Identifier for the block
        @type block_id: str
        """
        self.attributes = None
        super(ConvBlock, self).__init__(block_id)

    def get_in_channels(self) -> int:
        return self.modules[0].in_channels

    def get_out_channels(self) -> int:
        return self.modules[0].out_channels

    def is_deconvolution_enabled(self) -> bool:
        return self.attributes is not None

    def transpose(self, extra: Optional[nn.Module] = None) -> Any:
        raise ConvException('Cannot transpose abstract Convolutional block')

    def get_conv_output_size(self) -> ConvOutputSize:
        raise ConvException('Cannot transpose abstract Convolutional block')

    def __str__(self) -> AnyStr:
        modules_str = self.__repr__()
        return f'\n{self.block_id}:\nModules:\n{modules_str}'

    def __repr__(self) -> AnyStr:
        return '\n'.join([f'{idx}: {str(module)}' for idx, module in enumerate(self.modules)])

    def get_modules_weights(self) -> List[Any]:
        """
        Get the weights for modules which contains them
        @returns: weight of convolutional neural_blocks
        @rtype: tuple
        """
        return [module.weight.data for module in self.modules
                if type(module) is nn.Linear or type(module) is nn.Conv2d or type(module) is nn.Conv1d]
