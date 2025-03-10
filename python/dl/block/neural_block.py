__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch import nn
from typing import Self, AnyStr, Optional, List
from dl import DLException


"""
Basic Neural block for all deep learning architectures
"""


class NeuralBlock(nn.Module):
    supported_activations = ('Sigmoid', 'ReLU', 'Softmax', 'Tanh', 'ELU', 'LeakyReLU')

    def __init__(self, block_id: AnyStr, modules: List[nn.Module]):
        """
        Constructor for basic Neural block
        @param block_id: Optional identifier for the Neural block
        @type block_id: str
        @param modules: List of Torch modules
        @type modules: List[torch modules]
        """
        assert len(modules) > 0, f'Cannot create a Neural block without torch module'

        super(NeuralBlock, self).__init__()
        self.modules = modules
        self.block_id = block_id

    def transpose(self, extra: Optional[nn.Module] = None) -> Self:
        raise DLException('Cannot invert abstract Neural block')

    def __str__(self) -> AnyStr:
        module_repr = self.__repr__()
        return f'\n{self.block_id}\n{module_repr}'

    def __repr__(self):
        return '\n'.join([f'{idx}: {str(module)}' for idx, module in enumerate(self.modules)])

