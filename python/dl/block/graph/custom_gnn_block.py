__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from dl.block.neural_block import NeuralBlock
from typing import AnyStr, List
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing

"""
Implementation of a very simple Graph Convolutional Neural block which consists of 
- Message passing operator
- Optional activation function
- Optional batch norm 1-dimension
- Optional drop-out
"""


class CustomGNNBlock(NeuralBlock):
    def __init__(self,
                 _id: AnyStr,
                 message_passing: MessagePassing,
                 activation: nn.Module = None,
                 batch_norm: nn.BatchNorm1d = None,
                 drop_out: float = 0.0) -> None:
        """
        Constructor for this simple Graph Neural block
        @param _id: Identifier for the Graph neural block
        @type _id: str
        @param message_passing: Message passing operator (Conv,....)
        @type message_passing: nn.conv.MessagePassing
        @param activation: Activation function if defined
        @type activation: nn.Module
        @param batch_norm: 1-dimension batch norm
        @type batch_norm: BatchNorm1d
        @param drop_out: Drop out value is defined as 0.1 <= drop_out <= 0.9
        @type drop_out: float
        """
        assert drop_out == 0 or 0.1 <= drop_out < 0.9, f'Drop out is {drop_out} is out of range ]0.1, 0.9['
        self.id = _id

        modules: List[nn.Module] = [message_passing]
        if batch_norm is not None:
            modules.append(batch_norm)
        if drop_out > 0.0:
            modules.append(nn.Dropout(drop_out))
        if activation is not None:
            modules.append(activation)

        super(CustomGNNBlock, self).__init__(_id, tuple(modules))

    def __repr__(self) -> AnyStr:
        modules_str = '\n'.join([str(module) for module in self.modules])
        return f'\nGCN Modules:\n{modules_str}'
