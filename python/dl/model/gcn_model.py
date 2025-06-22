__author__ = "Patrick Nicolas"
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
