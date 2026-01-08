__author__ = "Patrick Nicolas"
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

# 3rd Party library imports
import torch


class FloydWarshall(object):
    def __init__(self, edge_index: torch.Tensor, num_nodes: int) -> None:
        self.edge_index = edge_index
        self.num_nodes = num_nodes

    def __call__(self):
        # Initialize distance matrix with a large value
        dist = torch.full((self.num_nodes, self.num_nodes), float('inf')).to(self.edge_index.device)
        dist.fill_diagonal_(0)

        # Fill in immediate neighbors from edge_index
        dist[self.edge_index[0], self.edge_index[1]] = 1

        # Iterative update of the distances as
        # d[k+1] = min( d_ij[k], d_ik[k] _ d_kj[k]
        for k in range(self.num_nodes):
            dist = torch.min(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))
        return dist

