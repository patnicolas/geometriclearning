__author__ = "Patrick R. Nicolas"
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

from typing import Self, AnyStr, List
import toponetx as tnx
import numpy as np
import torch
from topology.simplicial_laplacian import SimplicialLaplacian


class SimplicialFeatureSet(object):
    """
    Implementation of the Simplicial Complex with operators and a feature set (embedded vector).
    The functionality is:
    - Computation of incidence and adjacency matrices
    - Computation of various Laplacian operators
    - Visualization of Simplicial Complexes
    """
    triangle_colors = ['blue', 'red', 'green', 'purple', 'grey', 'orange']
    tetrahedron_color = 'lightgrey'

    def __init__(self, feature_set: np.array, edge_set: List[List[int]], face_set: List[List[int]]) -> None:
        """
        Constructor for the Simplicial Complex Model. Shape of Numpy array for the edge and face sets
        are evaluated for consistency.
        
        @param feature_set: Feature set or feature vector 
        @type feature_set: Numpy array
        @param edge_set:  Edge set an array of pair of node indices
        @type edge_set: Numpy array
        @param face_set:  Face set as an array of 3 node indices
        @type face_set: Numpy array
        """
        # Validate the shape of indices of the simplicial complex
        SimplicialFeatureSet.__validate(edge_set, face_set)
        
        self.feature_set = feature_set
        # Tuple (Src -> Dest)
        self.edge_set = edge_set
        # Either triangle {x, y, z] or Tetrahedron {x, y, z, t}
        self.face_set = face_set
        self.simplicial_indices = self.edge_set + self.face_set

    @classmethod
    def build(cls, dimension: int, edge_set: List[List[int]], face_set: List[List[int]]) -> Self:
        """
        Alternative constructor for the Simplicial model that uses random value for features set. The size of the
        feature set matrix is computed from the list of edges node indices.
        The feature set is the matrix number of nodes x dimension as follows:
                Feature#1   Feature#2  ...  Feature#dimension
        Node 1
        Node 2

        @param dimension: Size of the feature vectors
        @type dimension: int
        @param edge_set: Edge set as a tensor of pair of node indices
        @type edge_set: Torch tensor
        @param face_set:  Face set as a tensor of tensor with 3 node indices
        @type face_set: Torch tensor
        @return: Instance of Simplicial model
        @rtype: SimplicialFeatureSet
        """
        import itertools
        assert dimension > 0, f'Dimension of random vector {dimension} should be > 0'

        num_nodes = max(list(itertools.chain.from_iterable(edge_set)))
        random_feature_set = torch.rand(num_nodes, dimension)
        return cls(random_feature_set, edge_set, face_set)

    def __str__(self) -> AnyStr:
        return f'\nFeatures:\n{self.feature_set}\nEdges:\n{self.edge_set}\nFaces:\n{self.face_set}'

    def adjacency_matrix(self, directed_graph: bool = False) -> np.array:
        """
        Computation of the adjacency matrix (edges - nodes)
        @param directed_graph: Flag that specify if the graph is directed or not (Default Undirected graph)
        @type directed_graph: bool
        @return: Adjacency matrix as a dense matrix
        @rtype: Numpy array
        """
        # Initialize adjacency matrix
        n = len(self.feature_set)
        A = np.zeros((n, n), dtype=int)

        # Fill in edges
        for u, v in self.edge_set:
            A[u-1, v-1] = 1
            if directed_graph:
                A[v-1, u-1] = 1
        return A

    def incidence_matrix(self, rank: int = 1, directed_graph: bool = True) -> np.array:
        """
        Extract the incidence matrix for a given rank and directed/undirected graph
        @param rank: Rank of the Simplicial complex
        @type rank: int
        @param directed_graph: Flag that specifies if the graph is directed
        @type directed_graph: bool
        @return: Incidence matrix 
        @rtype: Numpy array
        """
        assert 0 <= rank < 3, f'Rank of incidence matrix {rank} should be [0, 2]'

        sc = tnx.SimplicialComplex(self.simplicial_indices)
        _, _, incidence = sc.incidence_matrix(rank=rank, index=True, signed=directed_graph)
        return incidence.todense()

    def laplacian(self, simplicial_laplacian: SimplicialLaplacian) -> np.array:
        return simplicial_laplacian(self.simplicial_indices)

    """ -------------------------  Private Supporting methods ------------------ """

    @staticmethod
    def __validate(edge_set: np.array, face_set: np.array) -> None:
        assert len(edge_set) > 0, 'Simplicial requires at least one edge'
        assert all(len(sublist) == 2 for sublist in edge_set), f'All elements of edge list should have 2 indices'

        assert len(face_set) > 0, 'Simplicial requires at least face'
        assert all(len(sublist) in (3, 4) for sublist in face_set), \
            f'All elements of edge list should have 3 or 4 indices'
