__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch import nn
from typing import Self, AnyStr, Optional, Tuple
from dl import DLException


"""
Basic Neural block for all deep learning architectures
"""


class NeuralBlock(nn.Module):
    supported_activations = ('Sigmoid', 'ReLU', 'Softmax', 'Tanh', 'ELU', 'LeakyReLU')

    def __init__(self, block_id: AnyStr, modules: Tuple[nn.Module]):
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

    def __repr__(self):
        conf_repr = ' '.join([f'{str(module)}' for module in self.modules])
        block_id_str = self.block_id if len(self.block_id) > 0 else ''
        return f'Block: {block_id_str} - {conf_repr}'
