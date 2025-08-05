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
from typing import AnyStr, List, Optional
# 3rd Party imports
import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
# Library imports
from deeplearning.block.neural_block import NeuralBlock
__all__ = ['MessagePassingBlock']


class MessagePassingBlock(NeuralBlock):
    """
    Implementation of a very simple Graph Convolutional Neural block which consists of
    - Message passing operator
    - Optional activation function
    - Optional batch norm 1-dimension
    - Optional drop-out
    """
    def __init__(self,
                 block_id: AnyStr,
                 message_passing_module: MessagePassing,
                 batch_norm_module: Optional[nn.Module] = None,
                 activation_module: Optional[nn.Module] = None,
                 dropout_module: Optional[nn.Module] = None) -> None:
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
        @param dropout_module: Drop out for training
        @type dropout_module: nn.Module subclass
        """
        super(MessagePassingBlock, self).__init__(block_id)

        # Need at a minimum a message passing module
        modules: List[nn.Module] = [message_passing_module]
        # Optional batch normalization
        if batch_norm_module is not None:
            modules.append(batch_norm_module)
        # Optional activation
        if activation_module is not None:
            modules.append(activation_module)
        if dropout_module is not None:
            modules.append(dropout_module)
        self.modules_list = modules

    def forward(self,
                x: torch.Tensor,
                edge_index: Adj,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation along the network with an input x  an adjacency, edge_index and a batch

        @param x: Input tensor
        @type x: torch.Tensor
        @param edge_index: Adjacency matrix as an index pairs
        @type edge_index:
        @param batch: Batch
        @type batch: torch.Tensor
        @return: Output of the graph convolutional neural block
        @rtype: torch.Tensor
        """
        # The adjacency data is used in the first module
        conv_module = self.modules_list[0]
        x = conv_module(x, edge_index)

        # Process all the torch modules if defined
        for module in self.modules[1:]:
            x = module(x)
        return x

    def __repr__(self) -> AnyStr:
        modules_str = '\n'.join([str(module) for module in self.modules_list])
        return f'\nGraph Neural Network Modules:\n{modules_str}'
