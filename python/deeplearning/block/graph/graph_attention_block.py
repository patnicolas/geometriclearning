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

# Standard library imports
from typing import AnyStr, Optional, Dict, Any, Self, Generic, TypeVar

# 3rd Party Library Import
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv, RGATConv, AGNNConv, FusedGATConv, GPSConv, BatchNorm
from torch_geometric.typing import Adj
# Library imports
from deeplearning.block.graph.message_passing_block import MessagePassingBlock
__all__ = ['GraphAttentionBlock']

# Class types
# GATL for type of torch geometric GAT layer module
GATL = TypeVar('GATL')


class GraphAttentionBlock(MessagePassingBlock, Generic[GATL]):

    def __init__(self,
                 block_id: AnyStr,
                 graph_attention_layer: GATL,
                 batch_norm_module: Optional[BatchNorm] = None,
                 activation_module: Optional[nn.Module] = None,
                 dropout_module: Optional[nn.Dropout] = None) -> None:
        """
            Constructor for the Graph Attention Block

            @param block_id: Identifier for the Graph neural block
            @type block_id: str
            @param graph_attention_layer: Graph Attention Torch Module
            @type graph_attention_layer: nn.conv.MessagePassing subclasses GATConv, GATv2Conv, TransformerConv,
                    RGATConv, AGNNConv, FusedGATConv or GPSConv
            @param batch_norm_module: Generic batch norm
            @type batch_norm_module: BatchNorm subclass
            @param activation_module: Activation function if defined
            @type activation_module: nn.Module subclass
            @param dropout_module: Drop out for training
            @type dropout_module: nn.Module subclass
        """
        GraphAttentionBlock.__validate(graph_attention_layer, batch_norm_module)
        super(GraphAttentionBlock, self).__init__(block_id,
                                                  graph_attention_layer,
                                                  batch_norm_module,
                                                  activation_module,
                                                  dropout_module)

    @classmethod
    def build(cls, block_attributes: Dict[AnyStr, Any]) -> Self:
        """
        block_attributes = {
            'block_id': 'MyBlock',
            'attention_layer': GATConv(in_channels=num_node_features, out_channels=num_channels, heads),
            'activation': nn.ReLU(),
            'batch_norm': BatchNorm(num_channels),
            'dropout': 0.25
        }
        @param block_attributes: Attribute of this block
        @type block_attributes: Dictionary
        @return: Instance of GraphAttentionBlock
        @rtype: GraphAttentionBlock
        """
        GraphAttentionBlock.__validate_build(block_attributes)
        return cls(block_attributes['block_id'],
                   block_attributes['attention_layer'],
                   block_attributes['batch_norm'],
                   block_attributes['activation'],
                   nn.Dropout(block_attributes['dropout']))

    def forward(self,
                x: torch.Tensor,
                edge_index: Adj,
                batch: torch.Tensor = None) -> torch.Tensor:
        """
           Forward propagation along the network with an input x  an adjacency, edge_index and an optional batch
           @param x: Input tensor
           @type x: torch.Tensor
           @param edge_index: Adjacency matrix as an index pairs
           @type edge_index:
           @param batch: Batch
           @type batch: torch.Tensor
           @return: Output of the graph attention neural block
           @rtype: torch.Tensor
        """
        # Process all the torch modules if defined
        for module in self.modules_list:
            # Invokes the forward method with edge indices for the convolutional module
            if isinstance(module, (GATConv, GATv2Conv, TransformerConv, RGATConv, AGNNConv, FusedGATConv, GPSConv)):
                x = module(x, edge_index)
            # Invokes the forward method with edge indices and batch for the pooling modules
            else:
                x = module(x)
        return x

    def __str__(self) -> AnyStr:
        """
        Display the identifier and the list of Torch modules in this block
        """
        return f'\nId: {self.block_id}\nModules: {str(self.modules_list)}'

    """ --------------------------  Private Helper Methods ---------------------- 
        Validation of the input to the constructor
    """

    @staticmethod
    def __validate(graph_attention_layer: GATL, batch_norm_module: Optional[BatchNorm] = None) -> None:
        if not isinstance(graph_attention_layer,
                          (GATConv, GATv2Conv, TransformerConv, RGATConv, AGNNConv, FusedGATConv, GPSConv)):
            raise TypeError(f'Type of graph attention layer {type(graph_attention_layer)} is not supported')
        if batch_norm_module is not None and isinstance(type(batch_norm_module), BatchNorm):
            raise ValueError(f'batch norm type {type(batch_norm_module)} should be BatchNorm')

    @staticmethod
    def __validate_build(block_attributes: Dict[AnyStr, Any]) -> None:
        if (block_attributes['dropout'] is not None and
                (block_attributes['dropout'] < 0.0 or block_attributes['dropout'] > 0.5)):
            raise ValueError(f'dropout {block_attributes["dropout"]} should be [0., 0.5]')
