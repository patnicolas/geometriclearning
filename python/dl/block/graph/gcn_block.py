__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."


from dl.block.graph.g_message_passing_block import GMessagePassingBlock
from typing import AnyStr, Self, Optional, Dict, Any
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv


class GCNBlock(GMessagePassingBlock):
    def __int__(self,
                block_id: AnyStr,
                gcn_layer: GCNConv,
                batch_norm_module: Optional[BatchNorm] = None,
                activation_module: Optional[nn.Module] = None,
                drop_out_module: Optional[nn.Module] = None) -> None:
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
        @param drop_out_module: Drop out for training
        @type drop_out_module: nn.Module subclass
        """
        super(GCNBlock, self).__init__(block_id=block_id,
                                       message_passing_module=gcn_layer,
                                       batch_norm_module=batch_norm_module,
                                       activation_module=activation_module,
                                       graph_pooling_module=None,
                                       drop_out_module=drop_out_module)

    @classmethod
    def build(cls, block_attributes: Dict[AnyStr, Any])  -> Self:
        block_id = block_attributes['block_id']
        gcn_layer = block_attributes['message_passing']
        batch_norm_module = block_attributes['batch_norm']
        activation_module = block_attributes['activation']
        pooling_module = block_attributes['pooling']
        dropout_module = nn.Dropout(block_attributes['dropout_ratio']) if 0 < block_attributes['dropout_ratio'] < 1 \
            else None
        return cls(block_id=block_id,
                   message_passing_module=gcn_layer,
                   batch_norm_module=batch_norm_module,
                   activation_module=activation_module,
                   graph_pooling_module=pooling_module,
                   drop_out_module=dropout_module)




    @classmethod
    def build_from_params(cls,
                          _id: AnyStr,
                          input_layer_dim: int,
                          output_layer_dim: int,
                          activation: nn.Module,
                          drop_out: float = 0.0) -> Self:
        gcn_layer = GCNConv(input_layer_dim, output_layer_dim)
        batch_norm = BatchNorm(output_layer_dim)
        return cls(_id, gcn_layer, batch_norm, activation, drop_out)
