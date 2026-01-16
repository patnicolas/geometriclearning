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

# Standard Library imports
from typing import AnyStr, Self, Optional, Dict, Any
# 3rd Party imports
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv
# Library imports
from deeplearning.block.graph.message_passing_block import MessagePassingBlock
__all__ = ['GCNBlock']


class GCNBlock(MessagePassingBlock):
    """
    Neural block for generic Graph Neural Network

    A Neural block can be constructor directly from PyTorch modules (nn.Module) using the default constructor
    or from a descriptive dictionary of block attributes such as:
    {
        'block_id': 'my_model',
        'message_passing': nn.GraphConv,
        'batch_norm': nn.BatchNorm(64),
        'pooling': nn.TopKPooling
        'activation': nn.ReLU(),
        'dropout_ratio': 0.3
    }

    Reference: https://patricknicolas.substack.com/p/reusable-neural-blocks-in-pytorch
    """
    def __init__(self,
                 block_id: AnyStr,
                 gcn_layer: GCNConv,
                 batch_norm_module: Optional[BatchNorm] = None,
                 activation_module: Optional[nn.Module] = None,
                 dropout_module: Optional[nn.Module] = None) -> None:
        """
        Constructor for the base Graph Neural block
        @param block_id: Identifier for the Graph neural block
        @type block_id: str
        @param gcn_layer: Message passing operator (Conv,....)
        @type gcn_layer: nn.conv.GCNConv
        @param batch_norm_module: Generic batch norm
        @type batch_norm_module: BatchNorm subclass
        @param activation_module: Activation function if defined
        @type activation_module: nn.Module subclass
        @param dropout_module: Drop out for training
        @type dropout_module: nn.Module subclass
        """
        super(GCNBlock, self).__init__(block_id=block_id,
                                       message_passing_module=gcn_layer,
                                       batch_norm_module=batch_norm_module,
                                       activation_module=activation_module,
                                       dropout_module=dropout_module)

    @classmethod
    def build(cls, block_attributes: Dict[AnyStr, Any]) -> Self:
        """
        Alternative constructor that instantiates a generic Graph neural block from a dictionary
        of neural attributes
        @param block_attributes: Dictionary of neural attributes
        @type block_attributes: Dict[AnyStr, Any]
        @return: Instance of a GCNBlock
        @rtype: GCNBlock
        """
        block_id = block_attributes['block_id']
        gcn_layer = block_attributes['message_passing']
        batch_norm_module = block_attributes['batch_norm']
        activation_module = block_attributes['activation']
        dropout_module = nn.Dropout(block_attributes['dropout_ratio']) if 0 < block_attributes['dropout_ratio'] < 1 \
            else None

        return cls(block_id=block_id,
                   gcn_layer=gcn_layer,
                   batch_norm_module=batch_norm_module,
                   activation_module=activation_module,
                   dropout_module=dropout_module)

    @classmethod
    def build_from_params(cls,
                          block_id: AnyStr,
                          input_layer_dim: int,
                          output_layer_dim: int,
                          activation: nn.Module,
                          drop_out: float = 0.0) -> Self:
        """
        Alternative constructor using Graph convolutional layer parameters
        @param block_id: Identifier for the block
        @type block_id: str
        @param input_layer_dim: Size or dimension of the input layer
        @type input_layer_dim: int
        @param output_layer_dim: Size or dimension of the input layer
        @type output_layer_dim: int
        @param activation: Activation module
        @type activation: nn.Module
        @param drop_out: Drop-out regularization factor
        @type drop_out: float
        @return: Instance of GCNBlock
        @rtype: GCNBlock
        """
        gcn_layer = GCNConv(input_layer_dim, output_layer_dim)
        batch_norm = BatchNorm(output_layer_dim)
        return cls(block_id, gcn_layer, batch_norm, activation, nn.Dropout(drop_out))
