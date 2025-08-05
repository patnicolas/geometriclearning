__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

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
from typing import Self, AnyStr, Optional
# 3rd Party imports
from torch import nn
# Library imports
from deeplearning import MLPException
__all__ = ['NeuralBlock']


class NeuralBlock(nn.Module):
    """
    Basic Neural block for all deep learning architectures
    """
    supported_activations = ('Sigmoid', 'ReLU', 'Softmax', 'Tanh', 'ELU', 'LeakyReLU')

    def __init__(self, block_id: AnyStr):
        """
        Constructor for basic Neural block
        @param block_id: Optional identifier for the Neural block
        @type block_id: str
        """
        super(NeuralBlock, self).__init__()
        self.block_id = block_id

    def transpose(self, extra: Optional[nn.Module] = None) -> Self:
        raise MLPException('Cannot invert abstract Neural block')

    def __str__(self) -> AnyStr:
        module_repr = self.__repr__()
        return f'\n{self.block_id}\n{module_repr}'

    def __repr__(self):
        return '\n'.join([f'{idx}: {str(module)}' for idx, module in enumerate(self.modules)])

