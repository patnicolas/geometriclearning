__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."


from dl.block.graph.gnn_base_block import GNNBaseBlock
from typing import AnyStr
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv


class GCNBlock(GNNBaseBlock):
    def __int__(self,
                _id: AnyStr,
                input_layer_dim: int,
                output_layer_dim: int,
                activation: nn.Module,
                drop_out: float = 0.0) -> None:
        gcn_layer = GCNConv(input_layer_dim, output_layer_dim)
        super(GCNBlock, self).__init__(_id, gcn_layer, activation, BatchNorm(output_layer_dim), drop_out)