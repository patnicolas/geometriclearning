__author__ = "Patrick Nicolas"
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
from typing import List, AnyStr
# 3rd Party imports
from torch_geometric.data import Dataset
import networkx as nx
# Library imports
from topology.simplicial.graph_to_simplicial_complex import GraphToSimplicialComplex
from topology.simplicial.abstract_simplicial_complex import SimplicialElement, AbstractSimplicialComplex


class AbstractSimplicialComplexBuilder(object):
    """
    This class implements the generation of Features vector for each of the Simplicial elements (node, edges and faces)
    It leverage TopoNetX library
    Dataset:
    If the data set does not contain one type of the simplicial element it will generate these simplicial elements from
    the adjacency or incidence matrices and computation of the Hodge Laplacian for this type of element.
    """
    def __init__(self,
                 dataset: Dataset | AnyStr,
                 nx_graph: nx.Graph | None) -> None:
        self.dataset = dataset
        self.nx_graph = nx_graph
        self.simplicial_nodes = None
        self.simplicial_edges = None
        self.simplicial_faces = None

    def add_simplicial_nodes(self, simplicial_nodes: List[SimplicialElement]) -> None:
        self.simplicial_nodes = simplicial_nodes

    def add_simplicial_edges(self, simplicial_edges: List[SimplicialElement]) -> None:
        self.simplicial_edges = simplicial_edges

    def add_simplicial_faces(self, simplicial_faces: List[SimplicialElement]) -> None:
        self.simplicial_faces = simplicial_faces

    def __str__(self) -> AnyStr:
        return AbstractSimplicialComplex.to_string(self.simplicial_nodes, self.simplicial_edges, self.simplicial_faces)

    def __call__(self, num_eigenvectors: (int, int, int),  max_num_nodes_cliques: int) -> AbstractSimplicialComplex:
        """
            Method to convert a Graph into a simplicial complex with the following steps:
                1: Initialization of an undirected graph using NetworkX
                2: Add faces (triangles and Tetrahedrons) to the graph
                3: Add features with values from eigen decomposition to each node
                4: Add features with values from eigen decomposition to each edge
                5: Add features with values from eigen decomposition to each face

            Generate the Feature vectors for node, edges and faces of the simplicial complex. The features vector is
            inferred from the computation of the Hodge Laplacian if not available in the data set

            @param num_eigenvectors: List of number of eigenvectors for each of the type of simplicial elements (node,
                                    edge, face)
            @type num_eigenvectors: Tuple[int, int, int)
            @param max_num_nodes_cliques:  Maximum number of graph nodes for which the simplicial complex is extracted
                                        from cliques. The complex is extracted from the neighbors otherwise
            @type max_num_nodes_cliques: int
            @return: New Simplicial elements
            @rtype: AbstractSimplicialComplex
        """
        graph_to_simplicial = GraphToSimplicialComplex(self.nx_graph,
                                                       self.dataset,
                                                       max_num_nodes_cliques,
                                                       SimplexType.WithTriangles)
        tnx_complex = graph_to_simplicial.add_faces()

        # Generate the node, edge and face feature vectors using Hodge Laplacian
        node_feature_from_hodge_laplacian, edge_feature_from_hodge_laplacian, face_feature_from_hodge_laplacian = (
            GraphToSimplicialComplex.features_from_hodge_laplacian(tnx_complex, num_eigenvectors)
        )
        # Use the feature vector specified in the constructor for the graph nodes if provided (not None)
        # otherwise use the node element from the computation of the Hodge Laplacian
        node_simplicial_elements = node_feature_from_hodge_laplacian if self.simplicial_nodes is None \
            else self.simplicial_nodes

        # Use the feature vector specified in the constructor for the graph edges if provided (not None)
        # otherwise use the node element from the computation of the Hodge Laplacian
        edge_simplicial_elements = edge_feature_from_hodge_laplacian if self.simplicial_edges is None \
            else self.simplicial_edges

        # Use the feature vector specified in the constructor for the simplicial faces if provided (not None)
        # otherwise use the node element from the computation of the Hodge Laplacian
        face_simplicial_elements = face_feature_from_hodge_laplacian if self.simplicial_faces is None \
            else self.simplicial_faces

        return AbstractSimplicialComplex(node_simplicial_elements, edge_simplicial_elements, face_simplicial_elements)
