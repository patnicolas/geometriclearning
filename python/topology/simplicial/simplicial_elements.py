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


# Standard Library imports
from typing import Self, AnyStr, List, Optional
from dataclasses import dataclass
# 3rd Party imports
import toponetx as tnx
import numpy as np
import torch
# Library imports
from topology.simplicial.simplicial_laplacian import SimplicialLaplacian
__all__ = ['SimplicialElements', 'SimplicialElement']


@dataclass
class SimplicialElement:
    """
    Definition of the basic element of a Simplicial Complex {Node, Edge, Face} composed of
    - Feature vector
    - Indices of nodes defining this element

    @param node_indices: List of indices of nodes composing this simplicial element
    @type node_indices: List[int]
    @param feature_set: Feature vector or set associated with this simplicial element
    @type feature_set: Numpy array
    """
    node_indices: Optional[List[int]] = None
    feature_set: Optional[np.array] = None

    def __str__(self) -> AnyStr:
        output = []
        if self.feature_set is not None:
            output.append(str(self.feature_set))
        if self.node_indices is not None:
            output.append(', '.join([str(idx) for idx in self.node_indices]))
        return '\n'.join(output) if len(output) > 0 else ''


class SimplicialElements(object):
    """
    Implementation of the Simplicial Complex with operators and a feature set (embedded vector).
    The functionality is:
    - Computation of incidence and adjacency matrices
    - Computation of various Laplacian operators
    - Visualization of Simplicial Complexes
    """
    triangle_colors = ['blue', 'red', 'green', 'purple', 'grey', 'orange']
    tetrahedron_color = 'lightgrey'

    def __init__(self,
                 simplicial_nodes: List[SimplicialElement],
                 simplicial_edges: List[SimplicialElement],
                 simplicial_faces: List[SimplicialElement]) -> None:
        """
        Constructor for the Simplicial Complex Model. Shape of Numpy array for the edge and face sets
        are evaluated for consistency.
        
        @param simplicial_nodes: List of nodes elements
        @type simplicial_nodes:  List[SimplicialElement]
        @param simplicial_edges:  List of Edge elements
        @type simplicial_edges:  List[SimplicialElement]
        @param simplicial_faces:  List of Face elements
        @type simplicial_faces:  List[SimplicialElement]
        """
        # Validate the shape of indices of the simplicial complex
        SimplicialElements.__validate(simplicial_edges, simplicial_faces)
        
        self.simplicial_nodes = simplicial_nodes
        # Tuple (Src -> Dest)
        self.simplicial_edges = simplicial_edges
        # Either triangle {x, y, z] or Tetrahedron {x, y, z, t}
        self.simplicial_faces = simplicial_faces
        # Extract the
        edges_indices = [edge.node_indices for edge in simplicial_edges]
        faces_indices = [edge.node_indices for edge in simplicial_faces]
        self.simplicial_indices = edges_indices + faces_indices

    @classmethod
    def random(cls,
               node_feature_dimension: int,
               edge_node_indices: List[List[int]],
               face_node_indices: List[List[int]]) -> Self:
        """
        Alternative constructor for the Simplicial model that uses random value for node features set. The size of the
        feature set matrix is computed from the list of edges node indices.
        The feature set is the matrix number of nodes x dimension as follows:
                Feature#1   Feature#2  ...  Feature#dimension
        Node 1
        Node 2

        @param node_feature_dimension: Size of the feature vectors
        @type node_feature_dimension: int
        @param edge_node_indices: Edge set as a tensor of pair of node indices
        @type edge_node_indices: Torch tensor
        @param face_node_indices:  Face set as a tensor of tensor with 3 node indices
        @type face_node_indices: Torch tensor
        @return: Instance of Simplicial model
        @rtype: SimplicialElements
        """
        import itertools
        assert node_feature_dimension > 0, f'Dimension of random vector {node_feature_dimension} should be > 0'

        # Retrieve the number of nodes from the largest index in the edge indices list
        num_nodes = max(list(itertools.chain.from_iterable(edge_node_indices)))

        # Generate random feature vector for node
        random_feature_set = torch.rand(num_nodes, node_feature_dimension).numpy()
        # Build the simplicial nodes (the node indices are implicit)
        simplicial_nodes = [SimplicialElement(feature_set=feat) for feat in random_feature_set]
        # Build the simplicial edges with no feature vector
        simplicial_edges = [SimplicialElement(edge_idx) for edge_idx in edge_node_indices]
        # Build the simplicial faces with no feature vector
        simplicial_faces = [SimplicialElement(face_idx) for face_idx in face_node_indices]
        return cls(simplicial_nodes, simplicial_edges, simplicial_faces)

    def __str__(self) -> AnyStr:
        simplicial_nodes_str = [str(node) for node in self.simplicial_nodes]
        simplicial_edges_str = [str(edge) for edge in self.simplicial_edges]
        simplicial_faces_str = [str(face) for face in self.simplicial_faces]
        return f'\nNodes:\n{simplicial_nodes_str}\nEdges:\n{simplicial_edges_str}\nFaces:\n{simplicial_faces_str}'

    def adjacency_matrix(self, directed_graph: bool = False) -> np.array:
        """
        Computation of the adjacency matrix (edges - nodes)
        @param directed_graph: Flag that specify if the graph is directed or not (Default Undirected graph)
        @type directed_graph: bool
        @return: Adjacency matrix as a dense matrix
        @rtype: Numpy array
        """
        # Initialize adjacency matrix
        n = len(np.concatenate([node.feature_set for node in self.simplicial_nodes]))
        A = np.zeros((n, n), dtype=int)

        # Fill in edges
        for u, v in [edge.node_indices for edge in self.simplicial_edges]:
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
    def __validate(simplicial_edge: List[SimplicialElement], simplicial_face: List[SimplicialElement]) -> None:
        if simplicial_edge is not None:
            edge_set = [edge.node_indices for edge in simplicial_edge]
            assert len(edge_set) > 0, 'Simplicial requires at least one edge'
            assert all(len(sublist) == 2 for sublist in edge_set), f'All elements of edge list should have 2 indices'

        if simplicial_face is not None:
            face_set = [face.node_indices for face in simplicial_face]
            assert len(face_set) > 0, 'Simplicial requires at least face'
            assert all(len(sublist) in (3, 4) for sublist in face_set), \
                f'All elements of edge list should have 3 or 4 indices'
