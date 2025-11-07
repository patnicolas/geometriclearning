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
from typing import Self, AnyStr, List, Optional, Tuple, Dict
from dataclasses import dataclass
# 3rd Party imports
import toponetx as tnx
import numpy as np
import torch
# Library imports
from topology.complex_element import ComplexElement
from topology import TopologyException
from topology.simplicial.simplicial_laplacian import SimplicialLaplacian
from topology.graph_complex_elements import GraphComplexElements
__all__ = ['AbstractSimplicialComplex', 'SimplicialElement']


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
    node_indices: Tuple[int, ...] | None = None
    feature_set: Optional[np.array] = None

    def __call__(self, override_node_indices: Tuple[int, ...] | None = None) -> Tuple[Tuple, np.array] | None:
        """
        Generate a tuple (node indices, feature vector) for this specific element. The node indices list is
        overridden only if it has not been already defined.
        A topology exception is raised if the node indices to be returned is None

        @param override_node_indices: Optional node indices
        @type override_node_indices: List[int]
        @return: Tuple (node indices, feature vector)
        @rtype: Tuple[Tuple, np.array]
        """
        if self.node_indices is None and override_node_indices is not None:
            self.node_indices = override_node_indices
        if self.node_indices is None:
            raise TopologyException('No node indices has been defined for this simplicial element')

        return tuple(self.node_indices), self.feature_set

    def __str__(self) -> AnyStr:
        output = []
        if self.feature_set is not None:
            output.append(list(np.round(self.feature_set, 5)))
        if self.node_indices is not None:
            output.append(self.node_indices)
        return ", ".join(map(str, output)) if len(output) > 0 else ""


class AbstractSimplicialComplex(object):
    """
    Implementation of the Simplicial Complex with operators and a feature set (embedded vector).
    The functionality is:
    - Computation of incidence and adjacency matrices
    - Computation of various Laplacian operators
    - Visualization of Simplicial Complexes
    """
    triangle_colors = ['blue', 'red', 'green', 'purple', 'grey', 'orange']
    tetrahedron_color = 'lightgrey'

    def __init__(self, graph_complex_elements: GraphComplexElements) -> None:
        """
        Constructor for the Simplicial Complex Model. Shape of Numpy array for the edge and face sets
        are evaluated for consistency.
        
        @param graph_complex_elements: Graph Complex elements for Node, Edges and Faces or cells
        @type graph_complex_elements:  GraphComplexElements
        """
        # Validate the shape of indices of the simplicial complex
        AbstractSimplicialComplex.__validate(graph_complex_elements)
        self.graph_complex_elements = graph_complex_elements
        # Extract the
        edges_indices = [edge.node_indices for edge in graph_complex_elements.complex_edges]
        faces_indices = [edge.node_indices for edge in graph_complex_elements.complex_simplex_2]
        self.simplicial_indices = edges_indices + faces_indices

    @classmethod
    def random(cls,
               node_feature_dimension: int,
               edge_node_indices: List[Tuple[int, ...]],
               face_node_indices: List[Tuple[int, ...]]) -> Self:
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
        @rtype: AbstractSimplicialComplex
        """
        import itertools
        assert node_feature_dimension > 0, f'Dimension of random vector {node_feature_dimension} should be > 0'

        # Retrieve the number of nodes from the largest index in the edge indices list
        num_nodes = max(list(itertools.chain.from_iterable(edge_node_indices)))

        # Generate random feature vector for node
        random_feature_set = torch.rand(num_nodes, node_feature_dimension).numpy()
        # Build the simplicial nodes (the node indices are implicit)
        simplicial_nodes = [ComplexElement(feature_set=feat) for feat in random_feature_set]
        # Build the simplicial edges with no feature vector
        simplicial_edges = [ComplexElement(tuple(edge_idx)) for edge_idx in edge_node_indices]
        # Build the simplicial faces with no feature vector
        simplicial_faces = [ComplexElement(tuple(face_idx)) for face_idx in face_node_indices]
        return cls(GraphComplexElements(simplicial_nodes, simplicial_edges, simplicial_faces))

    def node_features_map(self) -> Dict[Tuple, np.array]:
        """
        Generate a dictionary/map for all nodes in this simplicial complex using the format
        { (1,): np.array([9.3, ...])}

        @return: Dictionary of Tuple of node indices - Feature vectors
        @rtype: Dict[Tuple, np.array]
        """
        simplicial_nodes = self.graph_complex_elements.complex_nodes
        node_indices = list(zip(list(range(len(simplicial_nodes)))))
        nodes_feature_set = [simplicial_node.feature_set for simplicial_node in simplicial_nodes]
        return dict(zip(node_indices, nodes_feature_set))

    def edge_features_map(self) -> Dict[Tuple, np.array]:
        """
        Generate a dictionary/map for all edge in this simplicial complex using the format
        { (1, 4): np.array([9.3, ...])}

        @return: Dictionary of Tuple of the 2 node indices  - Feature vectors
        @rtype: Dict[Tuple, np.array]
        """
        simplicial_edges = self.graph_complex_elements.complex_edges
        edges_node_indices = [tuple(simplicial_edge.node_indices) for simplicial_edge in simplicial_edges]
        edges_feature_set = [simplicial_edge.feature_set for simplicial_edge in simplicial_edges]
        return dict(zip(edges_node_indices, edges_feature_set))

    def face_features_map(self) -> Dict[Tuple, np.array]:
        """
        Generate a dictionary/map for all faces in this simplicial complex using the format
        { (1, 4, 11): np.array([9.3, ...])} or  { (1, 4, 11, 9): np.array([9.3, ...])}

        @return: Dictionary of Tuple of the 3 or 4 node indices  - Feature vectors
        @rtype: Dict[Tuple, np.array]
        """
        simplicial_faces =  self.graph_complex_elements.complex_simplex_2
        faces_node_indices = [tuple(simplicial_face.node_indices) for simplicial_face in simplicial_faces]
        faces_feature_set = [simplicial_face.feature_set for simplicial_face in simplicial_faces]
        return dict(zip(faces_node_indices, faces_feature_set))

    def __str__(self) -> AnyStr:
        return str(self.graph_complex_elements)

    def adjacency_matrix(self, directed_graph: bool = False) -> np.array:
        """
        Computation of the adjacency matrix (edges - nodes)
        @param directed_graph: Flag that specify if the graph is directed or not (Default Undirected graph)
        @type directed_graph: bool
        @return: Adjacency matrix as a dense matrix
        @rtype: Numpy array
        """
        # Initialize adjacency matrix
        n = len(np.concatenate([node.feature_set for node in self.graph_complex_elements.complex_nodes]))
        A = np.zeros((n, n), dtype=int)

        # Fill in edges
        for u, v in [edge.node_indices for edge in self.graph_complex_elements.complex_edges]:
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
    def __validate(graph_complex_elements: GraphComplexElements) -> None:
        simplicial_edge = graph_complex_elements.complex_edges
        if simplicial_edge is not None:
            edge_set = [edge.node_indices for edge in simplicial_edge]
            assert len(edge_set) > 0, 'Simplicial requires at least one edge'
            assert all(len(sublist) == 2 for sublist in edge_set), f'All elements of edge list should have 2 indices'

        simplicial_face = graph_complex_elements.complex_simplex_2
        if simplicial_face is not None:
            face_set = [face.node_indices for face in simplicial_face]
            assert len(face_set) > 0, 'Simplicial requires at least face'
            assert all(len(sublist) in (3, 4) for sublist in face_set), \
                f'All elements of edge list should have 3 or 4 indices'
