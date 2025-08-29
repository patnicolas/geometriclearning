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
from typing import AnyStr, Optional, Dict, Any, Self, Generic, TypeVar
# 3rd Party imports
import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GraphConv, GCNConv, GCN2Conv
from torch_geometric.nn.pool import TopKPooling, SAGPooling
from torch_geometric.typing import Adj
# Library imports
from deeplearning.block.graph.message_passing_block import MessagePassingBlock
__all__ = ['GraphConvBlock']

# Class types
CL = TypeVar('CL')
P = TypeVar('P')


class GraphConvBlock(MessagePassingBlock, Generic[CL, P]):
    def __init__(self,
                 block_id: AnyStr,
                 graph_conv_layer: CL,
                 batch_norm_module: Optional[BatchNorm] = None,
                 activation_module: Optional[nn.Module] = None,
                 pooling_module: Optional[P] = None,
                 dropout_module: Optional[nn.Dropout] = None) -> None:
        """
            Constructor for the Graph Convolutional Network

            @param block_id: Identifier for the Graph neural block
            @type block_id: str
            @param graph_conv_layer: Message passing operator (Conv,....)
            @type graph_conv_layer: nn.conv.MessagePassing
            @param batch_norm_module: Generic batch norm
            @type batch_norm_module: BatchNorm subclass
            @param activation_module: Activation function if defined
            @type activation_module: nn.Module subclass
            @param dropout_module: Drop out for training
            @type dropout_module: nn.Module subclass
            """
        if not isinstance(graph_conv_layer, (GraphConv, GCNConv, GCN2Conv)):
            raise TypeError(f'Type of graph convolutional layer {type(graph_conv_layer)} is not supported')
        if pooling_module is not None and not isinstance(pooling_module, (SAGPooling, TopKPooling)):
            raise TypeError(f'Type of pooling {type(pooling_module)} is not supported')

        super(GraphConvBlock, self).__init__(block_id,
                                             graph_conv_layer,
                                             batch_norm_module,
                                             activation_module,
                                             dropout_module)
        self.has_pooling = pooling_module is not None
        # Although it is not strictly required to have the dropout module be the last module,
        # it is recommended and therefore we re-order the modules in case both dropout and pooling are provided
        if self.has_pooling:
            if dropout_module is not None:
                self.modules_list.remove(dropout_module)
                self.modules_list.append(pooling_module)
                self.modules_list.append(dropout_module)
            else:
                self.modules_list.append(pooling_module)
        self.pooling_edge_index = None

    @classmethod
    def build(cls, block_attributes: Dict[AnyStr, Any]) -> Self:
        """
        block_attributes = {
            'block_id': 'MyBlock',
            'conv_layer': GraphConv(in_channels=num_node_features, out_channels=num_channels),
            'num_channels': num_channels,
            'activation': nn.ReLU(),
            'batch_norm': BatchNorm(num_channels),
            'pooling': None,
            'dropout': 0.25
        }
        @param block_attributes: Attribute of this block
        @type block_attributes: Dictionary
        @return: Instance of GConvBlock
        @rtype: GraphConvBlock
        """
        GraphConvBlock.__validate(block_attributes)
        return cls(block_attributes['block_id'],
                   block_attributes['conv_layer'],
                   block_attributes['batch_norm'],
                   block_attributes['activation'],
                   block_attributes['pooling'],
                   nn.Dropout(block_attributes['dropout']))

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
        # Process all the torch modules if defined
        for module in self.modules_list:
            if isinstance(module, GraphConv):
                x = module(x, edge_index)
            elif isinstance(module, TopKPooling):
                x, self.pooling_edge_index, _, _, _, _ = module(x, edge_index, None, batch)
            else:
                x = module(x)
        return x

    def __str__(self) -> AnyStr:
        return '\n'.join([str(module) for module in self.modules_list])

    @staticmethod
    def __validate(block_attributes: Dict[AnyStr, Any]) -> None:
        if block_attributes['conv_layer'] is None or not isinstance(block_attributes['conv_layer'], GraphConv):
            raise ValueError(f'SAGE layer type {block_attributes["conv_layer"]} should be GraphConv')
        if block_attributes['batch_norm'] is not None and not isinstance(block_attributes['batch_norm'], BatchNorm):
            raise ValueError(f'batch norm type {block_attributes["batch_norm"]} should be BatchNorm')
        if block_attributes['pooling'] is not None and not (isinstance(block_attributes['pooling'], SAGPooling)
                                                            or isinstance(block_attributes['pooling'], TopKPooling)):
            raise ValueError(f'pooling {block_attributes["pooling"]} should be SAGPooling | TopKPooling')
        if (block_attributes['dropout'] is not None and
                (block_attributes['dropout'] < 0.0 or block_attributes['dropout'] > 0.5)):
            raise ValueError(f'dropout {block_attributes["dropout"]} should be [0., 0.5]')