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
# 3rd Party imports
import torch
# Library imports
__all__ = ['OlliverRicci']






class OlliverRicci(object):
    def __init__(self) -> None:
        print('Ctr')

    @staticmethod
    def floyd_warshall(edge_index: torch.Tensor, num_nodes):
        # Initialize distance matrix with a large value
        dist = torch.full((num_nodes, num_nodes), float('inf')).to(edge_index.device)
        dist.fill_diagonal_(0)

        # Fill in immediate neighbors from edge_index
        dist[edge_index[0], edge_index[1]] = 1

        # Iterative update (Floyd-Warshall)
        for k in range(num_nodes):
            dist = torch.min(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))
        return dist
