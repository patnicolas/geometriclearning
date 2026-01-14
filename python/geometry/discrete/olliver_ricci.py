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
from typing import List, Tuple, Optional, Callable
import logging
import python
# 3rd Party imports
import torch
# Library imports
from geometry.discrete.floyd_warshall import FloydWarshall
from geometry.discrete.sinkhorn_knopp import SinkhornKnopp

__all__ = ['OlliverRicci']


class OlliverRicci(FloydWarshall):

    def __init__(self,
                 edge_index: List[Tuple[int, int]],
                 weights: Optional[torch.Tensor],
                 epsilon: float,
                 rc: Tuple[torch.Tensor, torch.Tensor] = None) -> None:
        super(OlliverRicci, self).__init__(edge_index=edge_index, is_undirected=True, weights=weights)
        self.adjacency = FloydWarshall.create_adjacency(edge_index=edge_index, is_indirect=True)
        (r, c) = rc if rc is not None else OlliverRicci.__get_marginal_distributions(self.adjacency)
        self.wasserstein_1_approximation = SinkhornKnopp.build(r, c, self, epsilon)

    @classmethod
    def build(cls,
              edge_index: List[Tuple[int, int]],
              geodesic_distance: Callable[[int], torch.Tensor],
              epsilon: float,
              rc: Tuple[torch.Tensor, torch.Tensor] = None):
        weights = geodesic_distance(len(edge_index))
        return cls(edge_index, weights, epsilon, rc)

    def curvature(self, n_iters: int, early_stop_threshold: float) -> torch.Tensor:
        """
        adj_matrix: (N, N) tensor
        shortest_paths: (N, N) tensor of geodesic distances
        alpha: mass to stay at the current node
        """
        curvature = torch.zeros_like(self.adjacency)
        # Load the shortest paths as the cost matrix in the Wasserstein distance
        shortest_paths = self.wasserstein_1_approximation.cost_matrix

        edges = torch.nonzero(self.adjacency)
        for u, v in edges:
            # Compute the approximate Wasserstein distance - Numerator
            num_iters, w1 = self.wasserstein_1_approximation(n_iters, early_stop_threshold)
            # Load the all-pairs shortest path between u and v nodes
            shortest_path_uv = shortest_paths[u, v]
            # Apply the Olliver-Ricci formula
            curvature[u, v] = 1 - (w1 / shortest_path_uv)
            if self.is_undirected:
                curvature[v, u] = curvature[u, v]
        return curvature

    """ -------------------------  Private Helper Methods -------------------------  """

    @staticmethod
    def __get_marginal_distributions(adjacency: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        joint_probability_measures = OlliverRicci.__compute_prob_measures(adjacency)
        # Extract the marginal distribution from the joint distribution
        r = joint_probability_measures.sum(dim=1)
        c = joint_probability_measures.sum(dim=0)
        return r, c

    @staticmethod
    def __compute_prob_measures(adjacency: torch.Tensor, alpha: float = 0.4) -> torch.Tensor:
        n = adjacency.shape[0]
        # Define neighborhood distributions m_x
        # m_x(v) = alpha if v=x, else (1-alpha)/degree if v is neighbor
        degrees = adjacency.sum(dim=1)
        eye = torch.eye(n)

        # Probability measures for all nodes: (N, N)
        probs = (alpha * eye) + ((1 - alpha) * adjacency / degrees.unsqueeze(1))
        return probs
