__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import AnyStr, Optional
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

    """
    Forward propagation along the network with an input x
    and an adjacency, edge_index
    """
    def forward(self,
                x: torch.Tensor,
                edge_index: Adj,
                batch: torch.Tensor) -> torch.Tensor:

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
