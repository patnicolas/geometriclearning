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
# 3rd Party imports
import torch
# Library imports
from geometry.discrete.floyd_warshall import FloydWarshall
from geometry.discrete.sinkhorn_knopp import SinkhornKnopp

__all__ = ['OlliverRicci']


class OlliverRicci(FloydWarshall):
    """
    Implementation of the computation of the Olliver-Ricci Curvature

    Over-squashing and over-smoothing represent the primary obstacles in training Graph Neural Networks (GNNs). While
    over-squashing generates critical information bottlenecks, over-smoothing leads to the excessive generalization of
    node features. By utilizing Ollivier-Ricci curvature, researchers can identify and remediate graph regions that
    obstruct effective information flow.
    Applied to graphs, Ollivier-Ricci Curvature (ORC) serves as a discrete approximation that uncovers local topology
    and geometric properties through the lens of optimal transport.

    Let G be a graph with a shortest-path metric (a.k.a. cost matrix) d, mu the probability measure on graph for a
    given node v. The Ollivier-Ricci curvature of a pair of node (i, j) is computed from the Wasserstein distance W
    .. math::
        \kappa_{OR}(i, j)= 1-\frac{W_{1}(\mu _{i}, \mu _{j})}{d(i, j)} \    \    \     \    [1]


    Reference Article: Curvature-informed Graph Learning https://patricknicolas.substack.com/publish/post/181931881

    This class has two constructors:
    __init__:  Default for which the user provides optionally the weights of the edges of the graph
    build: Alternative constructor that generate the edge weights from the closed form geodesic distance between two
        nodes laying into the underlying manifold.
    """
    __slots__ = ['adjacency', 'wasserstein_1_approximation']

    def __init__(self,
                 edge_index: List[Tuple[int, int]],
                 weights: Optional[torch.Tensor],
                 epsilon: float,
                 rc: Tuple[torch.Tensor, torch.Tensor] = None) -> None:
        """
        Constructor for the Olliver-Ricci curvature. It is assumed that the graph is undirected.

        @param edge_index: List of pairs (tuples) (index source node, index destination node)
        @type edge_index: Tuple[int, int]
        @param weights: Optional weights associated with the weights
        @type weights: torch.Tensor
        @param epsilon: Entropy regularization scale factor
        @type epsilon: float
        @param rc: Pair of marginal distributions for rows (r) and columns (c) of the joint distribution matrix used
                    for the Wasserstein distance
        @type rc: Tuple[Tensor, Tensor]
        """
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
        """
        Alternative constructor for the computation of the Olliver-Ricci curvature of a mesh or a graph. Contrary
        to the default constructor, this method take a closed-form of the geodesic distance on the underlying
        manifold and generate the weights for each edge.

        @param edge_index:  List of pairs (tuples) (index source node, index destination node)
        @type edge_index: Tuple[int, int]
        @param geodesic_distance: Closed formula for the geodesic distance of the underlying manifold
        @type geodesic_distance: Callable[[int], torch.Tensor]
        @param epsilon: Entropy regularization scale factor
        @type epsilon: float
        @param rc: Pair of marginal distributions for rows (r) and columns (c) of the joint distribution matrix used
                    for the Wasserstein distance
        @type rc: Tuple[Tensor, Tensor]
        @return: Instance of this class
        @rtype: OlliverRicci
        """
        weights = geodesic_distance(len(edge_index))
        return cls(edge_index, weights, epsilon, rc)

    def curvature(self, n_iters: int, early_stop_threshold: float) -> torch.Tensor:
        """
        Method that compute the curvature of a graph or mesh using the Olliver-Ricci formula:
            K = 1 - W/d
        W; Approximate 1-dimensional Wasserstein distance using the iterative Sinkhorn-Knopp algorithm
        d: Distance of the shortest path between any given nodes using the Floyd_Warshall formula

        @param n_iters: Maximum number of iterations allowed
        @type n_iters: int
        @param early_stop_threshold: Early stopping condition
        @type early_stop_threshold: float
        @return: Discrete curvature
        @rtype: torch.Tensor
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
