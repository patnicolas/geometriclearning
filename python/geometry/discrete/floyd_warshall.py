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
from typing import AnyStr, List, Tuple, Optional


class FloydWarshall(object):
    INF = float('inf')

    def __init__(self, edge_index: List[Tuple[int, int]], weights: Optional[torch.Tensor] = None) -> None:
        if weights is not None and len(edge_index) != len(weights):
            raise ValueError(f'Number of edges {len(edge_index)} differs from weights {len(weights.size())}')

        self.edge_index = edge_index
        self.weights = torch.ones(len(edge_index)) if weights is None else weights

    def __str__(self) -> AnyStr:
        return f'\nEdge indices:\n{str(self.edge_index)}\nWeights:{self.weights}'

    def __call__(self) -> torch.Tensor:
        # Extract the number of nodes or vertices
        max_index = max(sum(self.edge_index, ()))
        num_nodes = max_index + 1

        shortest_distances = self.__init_shortest_distances()

        # Iterative update of the distances as
        # d[k+1] = min( d_ij[k], d_ik[k] _ d_kj[k]
        for k in range(num_nodes):
            shortest_distances = torch.min(
                shortest_distances,
                shortest_distances[:, k].unsqueeze(1) + shortest_distances[k, :].unsqueeze(0)
            )
        return shortest_distances

    """ -------------------  Private Helper Methods --------------------  """

    def __init_shortest_distances(self) -> torch.Tensor:
        # Initialize the shortest distance tensor as infinite for non-diagonal values
        # and 0 for diagonal values
        num_edges = len(self.edge_index)
        distances = torch.full(size=(num_edges, num_edges), fill_value=FloydWarshall.INF)
        distances.fill_diagonal_(0)
        # Apply weights
        for idx, (i, j) in enumerate(self.edge_index):
            distances[i][j] = self.weights[idx]
        return distances



