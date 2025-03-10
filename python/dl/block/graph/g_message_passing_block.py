__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from dl.block.neural_block import NeuralBlock
from typing import AnyStr, List, Optional
import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

"""
Implementation of a very simple Graph Convolutional Neural block which consists of 
- Message passing operator
- Optional activation function
- Optional batch norm 1-dimension
- Optional drop-out
"""


class GMessagePassingBlock(NeuralBlock):
    def __init__(self,
                 block_id: AnyStr,
                 message_passing_module: MessagePassing,
                 batch_norm_module: Optional[nn.Module] = None,
                 activation_module: Optional[nn.Module] = None,
                 graph_pooling_module: Optional[nn.Module] = None,
                 drop_out_module: Optional[nn.Module] = None) -> None:
        """
        Constructor for the base Graph Neural block
        @param block_id: Identifier for the Graph neural block
        @type block_id: str
        @param message_passing_module: Message passing operator (Conv,....)
        @type message_passing_module: nn.conv.MessagePassing
        @param batch_norm_module: Generic batch norm
        @type batch_norm_module: BatchNorm subclass
        @param activation_module: Activation function if defined
        @type activation_module: nn.Module subclass
        @param graph_pooling_module: Graph Pooling module
        @type graph_pooling_module: nn.Module subclass
        @param drop_out_module: Drop out for training
        @type drop_out_module: nn.Module subclass
        """
        # Need at a minimum a message passing module
        modules: List[nn.Module] = [message_passing_module]
        # Optional batch normalization
        if batch_norm_module is not None:
            modules.append(batch_norm_module)
        # Optional activation
        if activation_module is not None:
            modules.append(activation_module)
        # Optional Graph pooling module
        if graph_pooling_module is not None:
            modules.append(graph_pooling_module)
        if drop_out_module is not None:
            modules.append(drop_out_module)
        super(GMessagePassingBlock, self).__init__(block_id, modules)

    def forward(self,
                x: torch.Tensor,
                edge_index: Adj) -> torch.Tensor:
        # The adjacency data is used in the first module
        conv_module = self.modules[0]
        x = conv_module(x, edge_index)

        # Process all the torch modules if defined
        for module in self.modules[1:]:
            x = module(x)
        return x

    def __repr__(self) -> AnyStr:
        modules_str = '\n'.join([str(module) for module in self.modules])
        return f'\nGCN Modules:\n{modules_str}'
