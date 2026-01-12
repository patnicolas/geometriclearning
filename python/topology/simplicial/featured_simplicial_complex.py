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
from typing import Self, AnyStr, List, Tuple, Dict, TypeVar
import itertools
# 3rd Party imports
import toponetx as tnx
import numpy as np
import torch
# Library imports
from topology.simplicial.featured_simplex import FeaturedSimplex
from topology.complex_laplacian import ComplexLaplacian
from topology.featured_complex import FeaturedComplex
__all__ = ['FeaturedSimplicialComplex']
T = TypeVar('T')


class FeaturedSimplicialComplex(FeaturedComplex[T]):
    """
    Implementation of the Simplicial Complex with operators and a feature set (embedded vector).
    The functionality is:
    - Generation of random simplicial complex
    - Computation of incidence and adjacency matrices
    - Computation of various Laplacian operators

    A simplicial complex is a graph with faces. It generalizes graphs that model higher-order relationships
    among data elements—not just pairwise (edges), but also triplets, quadruplets, and beyond (0-simplex: node,
    1-simplex: edge, 2-simplex: Triangle, 3-simplex: Tetrahedron,)
    Simplicial complex are usually associated with the analysis of shape in data, field known as Topological Data
    Analysis (TDA).

    The parameter type 'T' is the type of the first element of the representation of the components of this topological
    domain - Simplicial -> List[int],  Cell -> Cell, Hypergraph -> List[Tuple[int]]

    Reference Substack article: https://patricknicolas.substack.com/p/exploring-simplicial-complexes-for

    Note: The implementation of the adjacency matrix is specific to graph. The TopoNetX library has a generic
    adjacency matrix to support edges - edges and faces -faces.
    """
    def __init__(self, featured_simplices: frozenset[FeaturedSimplex]) -> None:
        """
        Constructor for the Simplicial Complex Model. Shape of Numpy array for the edge and face sets
        are evaluated for consistency.
        
        @param featured_simplices: Graph Complex elements for Node, Edges and Faces or cells
        @type featured_simplices:  FeaturedSimplicialElements
        """
        super(FeaturedSimplicialComplex, self).__init__()

        # Assign index for the featured nodes if it is not defined in the argument
        # of the constructor
        node_index = 1
        for featured_simplex in featured_simplices:
            if featured_simplex.simplex_indices is None:
                featured_simplex.simplex_indices = [node_index]
                node_index += 1

        self.featured_simplices = featured_simplices
        self._validate()
        # Store all node indices for all the simplices of this complex
        self.simplicial_indices = [simplex.simplex_indices for simplex in self.featured_simplices]
        # Instantiate the TopoNetX module
        self.simplicial_complex = tnx.SimplicialComplex(self.simplicial_indices)

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
        @rtype: FeaturedSimplicialComplex
        """
        if node_feature_dimension <= 0:
            raise ValueError(f'Dimension of random vector {node_feature_dimension} should be > 0')

        # Retrieve the number of nodes from the largest index in the edge indices list
        num_nodes = max(list(itertools.chain.from_iterable(edge_node_indices)))
        # Generate random feature vector for node
        random_feature_set = torch.rand(num_nodes, node_feature_dimension).numpy()
        # Build the simplicial nodes (the node indices are implicit)
        simplicial_nodes = [FeaturedSimplex(features=feat) for feat in random_feature_set]
        # Build the simplicial edges with no feature vector
        simplicial_edges = [FeaturedSimplex(tuple(edge_idx)) for edge_idx in edge_node_indices]
        # Build the simplicial faces with no feature vector
        simplicial_faces = [FeaturedSimplex(tuple(face_idx)) for face_idx in face_node_indices]
        return cls(frozenset(simplicial_nodes + simplicial_edges + simplicial_faces))

    def get_featured_simplices(self, rank: int) -> List[FeaturedSimplex]:
        """
        Retrieve the featured simplices for a given rank {0, 1 or 2}
        #param rank: Rank of the simplex
        @type rank: int
        @return: List of Features simplex for a given rank
        @rtype:  List[FeaturedSimplex]
        """
        if rank < 0 or rank > 2:
            raise ValueError(f'Simplex rank {rank}is out of bounds')
        return [simplex for simplex in self.featured_simplices if simplex.get_rank() == rank]

    def node_features_map(self) -> Dict[Tuple, np.array]:
        """
        Generate a dictionary/map for all nodes in this simplicial complex using the format
        { (1,): np.array([9.3, ...])}

        @return: Dictionary of Tuple of node indices - Feature vectors
        @rtype: Dict[Tuple, np.array]
        """
        simplicial_nodes = self.get_featured_simplices(rank=0)
        node_indices = list(zip(list(range(len(simplicial_nodes)))))
        nodes_feature_set = [simplicial_node.features for simplicial_node in simplicial_nodes]
        return dict(zip(node_indices, nodes_feature_set))

    def edge_features_map(self) -> Dict[Tuple, np.array]:
        """
        Generate a dictionary/map for all edge in this simplicial complex using the format
        { (1, 4): np.array([9.3, ...])}

        @return: Dictionary of Tuple of the 2 node indices  - Feature vectors
        @rtype: Dict[Tuple, np.array]
        """
        simplicial_edges = self.get_featured_simplices(rank=1)
        edges_node_indices = [tuple(simplicial_edge.simplex_indices) for simplicial_edge in simplicial_edges]
        edges_feature_set = [simplicial_edge.features for simplicial_edge in simplicial_edges]
        return dict(zip(edges_node_indices, edges_feature_set))

    def face_features_map(self) -> Dict[Tuple, np.array]:
        """
        Generate a dictionary/map for all faces in this simplicial complex using the format
        { (1, 4, 11): np.array([9.3, ...])} or  { (1, 4, 11, 9): np.array([9.3, ...])}

        @return: Dictionary of Tuple of the 3 or 4 node indices  - Feature vectors
        @rtype: Dict[Tuple, np.array]
        """
        simplicial_faces = self.get_featured_simplices(rank=2)
        faces_node_indices = [tuple(simplicial_face.simplex_indices) for simplicial_face in simplicial_faces]
        faces_feature_set = [simplicial_face.features for simplicial_face in simplicial_faces]
        return dict(zip(faces_node_indices, faces_feature_set))

    def __str__(self) -> AnyStr:
        return str(self.featured_simplices)

    def adjacency_matrix(self, rank: Tuple[int, int] | int, signed: bool = False) -> np.array:
        """
        Computation of the adjacency matrix nodes - nodes
        Note: TopoNetX library has a generic adjacency matrix to support edges - edges and
            faces - faces.
        @param rank: Rank of the Simplicial complex
        @type rank: int
        @param signed: Flag that specify if the graph is directed or not (Default Undirected graph)
        @type signed: bool
        @return: Adjacency matrix as a dense matrix
        @rtype: Numpy array
        """
        # Initialize adjacency matrix
        n = len(np.concatenate([node.features for node in self.get_featured_simplices(rank=0)]))
        A = np.zeros((n, n), dtype=int)

        # Fill in edges
        for u, v in [edge.simplex_indices for edge in self.get_featured_simplices(rank=1)]:
            A[u - 1, v - 1] = 1
            if signed:
                A[v - 1, u - 1] = 1
        return A

    def incidence_matrix(self, rank: Tuple[int, int] | int, signed: bool = True) -> np.array:
        """
        Extract the incidence matrix for a given rank and directed/undirected graph
        @param rank: Rank of the Simplicial complex
        @type rank: int
        @param signed: Flag that specifies if the graph is directed
        @type signed: bool
        @return: Incidence matrix 
        @rtype: Numpy array
        """
        if rank < 0 or rank > 2:
            raise ValueError(f'Rank of incidence matrix {rank} should be [0, 2]')

        sc = tnx.SimplicialComplex(self.simplicial_indices)
        _, _, incidence = sc.incidence_matrix(rank=rank, index=True, signed=signed)
        return incidence.todense()

    def laplacian(self, simplicial_laplacian: ComplexLaplacian) -> np.array:
        """
        Computation of Up, Down, and Hodge Laplacian for Simplicial complex or rank 0, 1, and 2. This method invokes
        ComplexLaplacian.__call__ method

        @param simplicial_laplacian: Instance of the Laplacian for simplicial complex
        @type simplicial_laplacian: ComplexLaplacian
        @return: Laplacian for this simplicial as 2-dimensional Numpy array
        @rtype: Numpy array
        """
        return simplicial_laplacian(self.simplicial_indices)

    """ -------------------------  Private Supporting methods ------------------ """

    def _validate(self) -> None:
        simplicial_edge = self.get_featured_simplices(rank=1)
        if simplicial_edge is not None:
            edge_set = [edge.simplex_indices for edge in simplicial_edge]
            assert len(edge_set) > 0, 'Simplicial requires at least one edge'
            assert all(len(sublist) == 2 for sublist in edge_set), f'All elements of edge list should have 2 indices'

        simplicial_face = self.get_featured_simplices(rank=2)
        if simplicial_face is not None:
            face_set = [face.simplex_indices for face in simplicial_face]
            assert len(face_set) > 0, 'Simplicial requires at least face'
            assert all(len(sublist) in (3, 4) for sublist in face_set), \
                f'All elements of edge list should have 3 or 4 indices'
