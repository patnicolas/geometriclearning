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
from typing import AnyStr, Dict, Self, List
import logging
from enum import IntEnum
# 3rd Party imports
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import toponetx as tnx
from toponetx.transform import graph_to_clique_complex, graph_to_neighbor_complex
import python
from topology.simplicial.simplicial_elements import SimplicialElement, SimplicialElements


class SimplexType(IntEnum):
    Nodes = 0
    WithEdges = 1
    WithTriangles = 2
    WithTetrahedrons = 3


class GraphToSimplicial(object):
    types_map = {1: 'nodes', 2: 'edges', 3: 'triangles', 4: 'tetrahedrons'}

    def __init__(self,
                 dataset_name: AnyStr,
                 data: Data,
                 threshold_clique_complex: int,
                 simplex_types: SimplexType) -> None:
        """
        Constructor for the Generator of simplicial complex from a Graph.

        @param dataset_name: Name of PyTorch Geometric data set
        @type dataset_name: str
        @param data: PyTorch Geometric
        @type data: Graph data (nodes, edge indices)
        @param threshold_clique_complex:  Maximum number of graph nodes for which the simplicial complex is extracted
                                        from cliques. The complex is extracted from the neighbors otherwise
        @type threshold_clique_complex: int
        @param simplex_types: Type of Simplices to be collected from the graph
        @type simplex_types: Enumerator
        """
        self.data = data
        # The maximum rank of the simplices in the graph is computed from the type of simplicial to generate
        self.max_rank = simplex_types.value
        self.dataset_name = dataset_name
        self.threshold_clique_complex = threshold_clique_complex

    def __call__(self,  num_eigenvectors: List[int]) -> SimplicialElements:
        graph = self.__initialize_networkx_graph()
        simplicial_complex = self.__generate_faces_indices(graph)
        return GraphToSimplicial.__generate_simplicial_elements(
            simplicial_complex,
            num_eigenvectors
        )

    @staticmethod
    def count_simplex_by_type(simplicial_complex: tnx.SimplicialComplex) -> Dict[AnyStr, int]:
        """
        Extract the number of nodes, edges and faces associated with this simplicial complex
        
        @param simplicial_complex: TopoX simplicial complex instance
        @type simplicial_complex: tnx.SimplicialComplex
        @return: Dictionary of simplices per type (node, edge, triangle and tetrahedron)
        @rtype: Dict
        """
        from itertools import groupby

        sorted_simplex = sorted(simplicial_complex.simplices, key=len)
        return {GraphToSimplicial.types_map[length]: sum(1 for _ in group)
                for length, group in groupby(sorted_simplex, key=len) if length < 5}

    """ ----------------  Private Helper Methods -------------------- """

    def __initialize_networkx_graph(self) -> nx.Graph:
        # Create a NetworkX graph
        G = nx.Graph()
        # Populate with the node from the dataset
        G.add_nodes_from(range(self.data.num_nodes))

        # Populate with the edges from the dataset: We need to transpose the tensor from 2 x num edges shape to
        # num edges x 2 shape
        edge_idx = self.data.edge_index.cpu().T
        G.add_edges_from(edge_idx.tolist())
        return G

    def __generate_faces_indices(self, G: nx.Graph) -> tnx.SimplicialComplex:
        # If faces have to be generated...
        if self.max_rank > 0:
            # If this is a smaller graph that we can extract the cliques directly
            if self.data.num_nodes < self.threshold_clique_complex:
                logging.info(f'Generate complex from cliques with {self.data.num_nodes} nodes')
                return graph_to_clique_complex(G, self.max_rank)
            else:
                logging.info(f'Generate complex from neighbors with {self.data.num_nodes} nodes')
                return graph_to_neighbor_complex(G)
        # Otherwise just generate the graph
        else:
            return tnx.SimplicialComplex(G)

    @staticmethod
    def __generate_simplicial_elements(cplx: tnx.SimplicialComplex,
                                       num_eigenvectors: List[int]) -> SimplicialElements:
        from toponetx.algorithms.spectrum import hodge_laplacian_eigenvectors

        simplicial_features = [hodge_laplacian_eigenvectors(cplx.hodge_laplacian_matrix(idx), num_eigenvectors[idx])[1]
                               for idx in range(len(num_eigenvectors))]
        simplicial_elements = [GraphToSimplicial.compute_simplicial_elements(cplx, simplicial_features, idx)
                               for idx in range(len(simplicial_features))]
        simplicial_faces = simplicial_elements[2] + simplicial_elements[3]

        return SimplicialElements(simplicial_elements[0], simplicial_elements[1], simplicial_faces)

    @staticmethod
    def compute_simplicial_elements(cplx: tnx.SimplicialComplex,
                                    simplicial_features: List,
                                    index: int) -> List[SimplicialElement]:
        simplicial_node_feat = zip(cplx.skeleton(index), np.array(simplicial_features[index]), strict=True)
        return [SimplicialElement(list(u), v) for u, v in simplicial_node_feat]

