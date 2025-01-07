__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from dl.model.gnn_base_model import GNNBaseModel
from dl.block.graph.gcn_block import GCNBlock
from typing import AnyStr, List


class GCNModel(GNNBaseModel):
    def __init__(self,
                 model_id: AnyStr,
                 batch_size: int,
                 walk_length: int,
                 gnn_blocks: List[GCNBlock]) -> None:
        super(GCNModel, self).__init__(model_id, batch_size, walk_length, gnn_blocks)
