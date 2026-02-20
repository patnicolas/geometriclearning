__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard Library imports
from typing import Any, AnyStr, Optional, List
# 3rd Party imports
from torch import nn
# Library imports
from deeplearning import ConvException
from deeplearning.block.neural_block import NeuralBlock
from deeplearning.block.conv.conv_output_size import ConvOutputSize
__all__ = ['ConvBlock']


class ConvBlock(NeuralBlock):
    """
    Generic convolutional neural block for 1, 2 and 3 imensions

    Formula to compute output_dim of a convolutional block given an in_channels
            output_dim = (in_channels + 2*padding - kernel_size)/stride + 1
    Note: Spectral Normalized convolution is available only for 2D models

    Reference: https://patricknicolas.substack.com/p/reusable-neural-blocks-in-pytorch
    """
    __slots__ = ['attributes']

    def __init__(self, block_id: Optional[AnyStr]) -> None:
        """
        Constructor for the Generic Convolutional Neural block
        @param block_id: Identifier for the block
        @type block_id: str
        """
        self.attributes = None
        super(ConvBlock, self).__init__(block_id)

    def get_in_channels(self) -> int:
        return self.modules_list[0].in_channels

    def get_out_channels(self) -> int:
        return self.modules_list[0].out_channels

    def is_deconvolution_enabled(self) -> bool:
        return self.model_attributes is not None

    def transpose(self, extra: Optional[nn.Module] = None) -> Any:
        raise ConvException('Cannot transpose abstract Convolutional block')

    def get_conv_output_size(self) -> ConvOutputSize:
        raise ConvException('Cannot transpose abstract Convolutional block')

    def __str__(self) -> AnyStr:
        modules_str = self.__repr__()
        return f'\n{self.block_id}:\nModules:\n{modules_str}'

    def __repr__(self) -> AnyStr:
        return '\n'.join([f'{idx}: {str(module)}' for idx, module in enumerate(self.modules_list)])

    def get_modules_weights(self) -> List[Any]:
        """
        Get the weights for modules which contains them
        @returns: weight of convolutional neural_blocks
        @rtype: tuple
        """
        return [module.weight.data_dict for module in self.modules_list
                if type(module) is nn.Linear or type(module) is nn.Conv2d or type(module) is nn.Conv1d]
