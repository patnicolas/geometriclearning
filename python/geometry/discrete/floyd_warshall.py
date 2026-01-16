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
    """
    The Floyd-Warshall (FW) algorithm is a commonly applied method to compute all-pairs shortest paths in a directed
    graph. The algorithm assumes that the weights are all positive.

    Algorithm:
    Let’s consider a subset S={1, 2, … k} subset of k vertices (or nodes) from a graph V = {1, 2, …, k, .., n}.
    For any pair of nodes i, j in V,  FW considers all the paths whose intermediate vertices are all down from S.
    Let p be a minimum weight path from among them. FW exploits the relationship between the path p and shortest
    path i → j with all intermediate nodes in the set {1, 2, …, k-1}

    .. math::
        d_{i\to j}^{(k)}=\begin{matrix}
        w_{ij}  & if \ k=0 \\
        min_{i,j,k} \left( d_{i\to j}^{(k-1)} , d_{i\to k}^{(k-1)} + d_{k\to j}^{(k-1)}\right) & if \  k \ge 1
        \end{matrix}y
    """
    INF = float('inf')

    def __init__(self,
                 edge_index: List[Tuple[int, int]],
                 weights: Optional[torch.Tensor] = None,
                 is_undirected: bool = True) -> None:
        """
        Default constructor for the Floyd-Warshall algorithm.
        Layout:   Edge = (index source vertex, index destination vertex) -> Weight
        Weight are set to 1.0 if not defined (default value None).
        
        @param edge_index: List of pairs (tuples) (index source node, index destination node)
        @type edge_index: Tuple[int, int]
        @param weights: Optional weights for each of the edge
        @type weights: torch.Tensor
        @param is_undirected Boolean flag to specify if this graph is undirected
        @type is_undirected bool
        """
        if weights is not None and len(edge_index) != len(weights):
            raise ValueError(f'Number of edges {len(edge_index)} differs from weights {len(weights.size())}')

        self.edge_index = edge_index
        self.is_undirected = is_undirected
        self.weights = torch.ones(len(edge_index))/len(edge_index) if weights is None else weights/weights.sum()

    def __str__(self) -> AnyStr:
        return f'\nEdge indices:\n{str(self.edge_index)}\nWeights:{self.weights}'

    @staticmethod
    def create_adjacency(edge_index: List[Tuple[int, int]], is_indirect: bool = True) -> torch.Tensor:
        n_nodes = max(sum(edge_index, ())) + 1
        adjacency = torch.zeros(n_nodes, n_nodes)
        for i, j in edge_index:
            adjacency[i][j] = 1
            if is_indirect:
                adjacency[j][i] = 1
        return adjacency

    def __call__(self) -> torch.Tensor:
        """
        Computation of the all-pairs shortest parts across all the nodes
        @return: Tensor of all pairs shortest distances
        @rtype: torch.Tensor
        """
        # Extract the number of nodes or vertices
        max_index = max(sum(self.edge_index, ()))
        num_nodes = max_index + 1

        # Initialize the shortest distances
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
        # Initialize the shortest distance tensor as infinite for non-diagonal values, 0 for diagonal values
        num_nodes = max(sum(self.edge_index, ()))+1
        distances = torch.full(size=(num_nodes, num_nodes), fill_value=FloydWarshall.INF)
        distances.fill_diagonal_(0)
        # Apply weights
        for idx, (i, j) in enumerate(self.edge_index):
            distances[i][j] = self.weights[idx]
            if self.is_undirected:
                distances[j][i] = distances[i][j]
        return distances



