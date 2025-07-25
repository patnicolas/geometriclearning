__author__ = "Patrick Nicolas"
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

from torch import nn
from typing import Self, AnyStr, Optional, List
from dl.block import MLPException
from abc import ABC
from abc import abstractmethod
__all__ = ['NeuralBlock']


class NeuralBlock(nn.Module, ABC):
    """
    Basic Neural block for all deep learning architectures. This class cannot be directly instantiated and
    requires sub-classing
    """
    # List of supported activation method
    supported_activations = ('Sigmoid', 'ReLU', 'Softmax', 'Tanh', 'ELU', 'LeakyReLU')

    def __init__(self, block_id: AnyStr) -> None:
        """
        Constructor for basic Neural block
        @param block_id: Optional identifier for the Neural block
        @type block_id: str
        """
        super(NeuralBlock, self).__init__()
        self.block_id = block_id

    @abstractmethod
    def transpose(self, activation_update: Optional[nn.Module] = None) -> Self:
        """
        Transpose this block from encoder to decoder
        Example: Auto-encoder, convolutional to de-convolutional network

        @param activation_update: Optional activation module to override the original one
        @type activation_update: nn.Module
        @return: Instance of this sub-class of Neural block
        @rtype: NeuralBlock
        """
        pass

    def __str__(self) -> AnyStr:
        module_repr = self.__repr__()
        return f'\n{self.block_id}\n{module_repr}'

    def __repr__(self):
        return '\n'.join([f'{idx}: {str(module)}' for idx, module in enumerate(self.modules)])

