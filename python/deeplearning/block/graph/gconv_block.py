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

from typing import AnyStr, Optional, Dict, Any, Self
import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GraphConv
from torch_geometric.nn.pool import TopKPooling, SAGPooling
from torch_geometric.typing import Adj


class GConvBlock(nn.Module):
    def __init__(self,
                 block_id: AnyStr,
                 gconv_layer: GraphConv,
                 batch_norm_module: Optional[BatchNorm] = None,
                 activation_module: Optional[nn.Module] = None,
                 pooling_module: Optional[SAGPooling | TopKPooling] = None,
                 dropout_module: Optional[nn.Dropout] = None) -> None:

        super(GConvBlock, self).__init__()
        self.block_id = block_id

        # Iteratively build the sequence of Torch Module according
        # to the order of the arguments of the constructor
        modules_list = nn.ModuleList()
        modules_list.append(gconv_layer)
        if batch_norm_module is not None:
            modules_list.append(batch_norm_module)
        if activation_module is not None:
            modules_list.append(activation_module)
        if pooling_module is not None:
            modules_list.append(pooling_module)
        if dropout_module is not None:
            modules_list.append(dropout_module)
        self.modules_list = modules_list

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
        @rtype: GConvBlock
        """
        block_id = block_attributes['block_id']
        conv_layer_attribute = block_attributes['conv_layer']
        activation_attribute = block_attributes['activation']
        batch_norm_attribute = block_attributes['batch_norm']
        pooling_attribute = block_attributes['pooling']
        dropout_attribute = block_attributes['dropout']
        return cls(block_id,
                   conv_layer_attribute,
                   batch_norm_attribute,
                   activation_attribute,
                   pooling_attribute,
                   nn.Dropout(dropout_attribute) if dropout_attribute > 0 else None)

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
                x, edge_index, _, _, _, _ = module(x, edge_index, None, batch)
            else:
                x = module(x)
        return x

    def __str__(self) -> AnyStr:
        return '\n'.join([str(module) for module in self.modules_list])
