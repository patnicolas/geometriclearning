__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import AnyStr, List, Optional
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
                 drop_out_module: Optional[nn.Dropout] = None) -> None:

        super(GConvBlock, self).__init__()
        self.block_id = block_id

        # Iteratively build the sequence of Torch Module according
        # to the order of the arguments of the constructor
        modules: List[nn.Module] = [gconv_layer]
        if batch_norm_module is not None:
            modules.append(batch_norm_module)
        if activation_module is not None:
            modules.append(activation_module)
        if pooling_module is not None:
            modules.append(pooling_module)
        if drop_out_module is not None:
            modules.append(drop_out_module)
        self.modules = modules

    """
    Forward propagation along the network with an input x
    and an adjacency, edge_index
    """
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

    def __str__(self) -> AnyStr:
        return '\n'.join([str(module) for module in self.modules])
