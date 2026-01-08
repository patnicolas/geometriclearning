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
from typing import AnyStr, List, Tuple, Self


class FloydWarshall(object):
    INF = float('inf')

    def __init__(self, edge_index: List[Tuple[int, int]], weights: torch.Tensor) -> None:
        if len(edge_index) != len(weights):
            raise ValueError(f'Number of edges {len(edge_index)} differs from weights {len(weights.size())}')

        # Extract the number of nodes or vertices
        max_index = max(sum(edge_index, ()))
        self.num_nodes = max_index + 1

        # Initialize the shortest distance tensor as infinite for non diagonal values
        # and 0 for diagonal values
        self.distances = torch.full(size=(self.num_nodes, self.num_nodes), fill_value=FloydWarshall.INF)
        self.distances.fill_diagonal_(0)
        # Apply weights
        for idx, (i, j) in enumerate(edge_index):
            self.distances[i][j] = weights[idx]

    @classmethod
    def build(cls, edge_index: List[Tuple[int, int]]) -> Self:
        num_nodes = max(sum(edge_index, ())) +1
        weights = torch.ones(num_nodes,  num_nodes)
        return cls(edge_index, weights)

    def __str__(self) -> AnyStr:
        return f'\n{str(self.distances)}'

    def __call__(self) -> None:
        # Iterative update of the distances as
        # d[k+1] = min( d_ij[k], d_ik[k] _ d_kj[k]
        for k in range(self.num_nodes):
            self.distances = torch.min(self.distances, self.distances[:, k].unsqueeze(1) + self.distances[k, :].unsqueeze(0))


