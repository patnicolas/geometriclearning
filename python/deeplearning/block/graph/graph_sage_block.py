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
from typing import AnyStr, Optional, Dict, Any, Self, TypeVar, Generic
# 3rd Party imports
import torch.nn as nn
from torch_geometric.typing import Adj
import torch
from torch_geometric.nn import BatchNorm, SAGEConv, CuGraphSAGEConv
# Library imports
from deeplearning.block.graph.message_passing_block import MessagePassingBlock
__all__ = ['GraphSAGEBlock']

CL = TypeVar('CL')


class GraphSAGEBlock(MessagePassingBlock, Generic[CL]):
    def __init__(self,
                 block_id: AnyStr,
                 graph_SAGE_layer: CL,
                 batch_norm_module: Optional[BatchNorm] = None,
                 activation_module: Optional[nn.Module] = None,
                 dropout_module: Optional[nn.Dropout] = None) -> None:
        """
        Default constructor that do not include validation of configuration parameters. For validation use
        the build method.
        
        @param block_id: Identifier for the block
        @type block_id: str
        @param graph_SAGE_layer: PyTorch Geometric module for SAGE model
        @type graph_SAGE_layer: Generic T
        @param batch_norm_module: Optional  PyTorch Geometric module for batch normalization
        @type batch_norm_module: Optional[BatchNorm]
        @param activation_module: Optional PyTorch Geometric module for activation
        @type activation_module: Optional[Module]
        @param dropout_module: Optional PyTorch Geometric module for regularization 
        @type dropout_module: Optional[Dropout]
        """
        if not isinstance(graph_SAGE_layer, (SAGEConv, CuGraphSAGEConv)):
            raise TypeError(f'Type of graph_SAGE_layer {type(graph_SAGE_layer )} is incorrect')

        super(GraphSAGEBlock, self).__init__(block_id,
                                             graph_SAGE_layer,
                                             batch_norm_module,
                                             activation_module,
                                             dropout_module)

    @classmethod
    def build(cls, block_attributes: Dict[AnyStr, Any]) -> Self:
        """
        Alternative constructor for a SAGE block. This constructor validates the various attributes.
        
        block_attributes = {
            'block_id': str - 'MyBlock',
            'sage_layer': SAGEConv - SAGEConv(in_channels=num_node_features, out_channels=num_channels),
            'activation': nn.Module -  nn.ReLU(),
            'batch_norm': nn.Module - BatchNorm(num_channels),
            'dropout': float - 0.25
        }
        Raise KeyError if configuration parameters are incorrect
              ValueError if value of configuration parameters are incorrect

        @param block_attributes: Attribute of this block
        @type block_attributes: Dictionary
        @return: Instance of GConvBlock
        @rtype: GraphConvBlock
        """
        GraphSAGEBlock.__validate(block_attributes)
        return cls(block_attributes['block_id'],
                   block_attributes['SAGE_layer'],
                   block_attributes['batch_norm'],
                   block_attributes['activation'],
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
            x = module(x, edge_index) if isinstance(module, (SAGEConv, CuGraphSAGEConv)) else module(x)
        return x

    def reset_parameters(self) -> None:
        self.modules_list[0].reset_parameters()

    @staticmethod
    def __validate(block_attributes: Dict[AnyStr, Any]) -> None:
        if block_attributes['SAGE_layer'] is None or not isinstance(block_attributes['SAGE_layer'], SAGEConv):
            raise ValueError(f'SAGE layer type {block_attributes["SAGE_layer"]} should be SAGEConv')
        if block_attributes['batch_norm'] is not None and not isinstance(block_attributes['batch_norm'], BatchNorm):
            raise ValueError(f'batch norm type { block_attributes["batch_norm"] } should be BatchNorm')
        if (block_attributes['dropout'] is not None and
                (block_attributes['dropout'] < 0.0 or block_attributes['dropout'] > 0.5)):
            raise ValueError(f'dropout {block_attributes["dropout"]} should be [0., 0.5]')

