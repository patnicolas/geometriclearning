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
from typing import AnyStr, Dict, List, AnyStr
import logging
from enum import IntEnum
# 3rd Party imports
from torch_geometric.data import Data, Dataset
import numpy as np
import networkx as nx
import toponetx as tnx
from toponetx.transform import graph_to_clique_complex, graph_to_neighbor_complex
# Library imports
import python
from topology.simplicial.abstract_simplicial_complex import SimplicialElement, AbstractSimplicialComplex
_all_ = ['SimplexType', 'GraphToSimplicial']


class SimplexType(IntEnum):
    Nodes = 0
    WithEdges = 1
    WithTriangles = 2
    WithTetrahedrons = 3


class GraphToSimplicial(object):
    types_map = {1: 'nodes', 2: 'edges', 3: 'triangles', 4: 'tetrahedrons'}
    """
    This class wraps the mechanism to convert a Graph into a Simplicial complex by adding faces and features for
    edges and faces in 5 steps:
            1: Initialization of an undirected graph using NetworkX
            2: Add faces (triangles and Tetrahedrons) to the graph
            3: Add features with values from eigen decomposition to each node
            4: Add features with values from eigen decomposition to each edge
            5: Add features with values from eigen decomposition to each face
    """
    def __init__(self,
                 nx_graph: nx.Graph | None,
                 dataset: AnyStr | Dataset,
                 max_num_nodes_cliques: int,
                 simplex_types: SimplexType) -> None:
        """
        Constructor for the Generator of simplicial complex from a Graph.
        A NetworkX can be optionally provided. If not it will be initialized by method initialize_networkx_graph
        A PyTorch Geometric sataset is provided either as a name or the dataset itself

        @param nx_graph: Optional NetworkX graph.
        @type nx_graph: Union[nx.Graph, None]
        @param dataset: Name of PyTorch Geometric data set OR dataset itself
        @type dataset: Union[str, Dataset]
        @param max_num_nodes_cliques:  Maximum number of graph nodes for which the simplicial complex is extracted
                                        from cliques. The complex is extracted from the neighbors otherwise
        @type max_num_nodes_cliques: int
        @param simplex_types: Type of Simplices to be collected from the graph
        @type simplex_types: Enumerator
        """
        assert max_num_nodes_cliques > 8, \
            f'Maximum number of nodes for which simplicial complex is generated from cliques'

        # If the data set is provided through its name
        if isinstance(dataset, str):
            from dataset.graph.pyg_datasets import PyGDatasets
            pyg_dataset = PyGDatasets(dataset)
            dataset = pyg_dataset()
            self.dataset_name = dataset
        # Otherwise extract the name as a dataset attribute
        else:
            self.dataset_name = getattr(dataset, 'name')

        self.nx_graph = GraphToSimplicial.__initialize_networkx_graph(nx_graph, dataset[0])

        self.data = dataset[0]
        # The maximum rank of the simplices in the graph is computed from the type of simplicial to generate
        self.max_rank = simplex_types.value
        self.threshold_clique_complex = max_num_nodes_cliques

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

    @staticmethod
    def __initialize_networkx_graph(G: nx.Graph, data: Data) -> nx.Graph:
        """
        STEP 1: Initialization of the graph if it has not been initialized.

        @param G: NetworkX graph if provided, None if not provided
        @type G: nx.Graph
        @param data: Graph data
        @type data: torch_geometric.data.Data
        @return: NetworkX undirected graph
        @rtype: Graph
        """
        if G is None:
            # Create a NetworkX graph
            G = nx.Graph()
            # Populate with the node from the dataset
            G.add_nodes_from(range(data.num_nodes))

            # Populate with the edges from the dataset: We need to transpose the tensor from 2 x num edges shape to
            # num edges x 2 shape
            edge_idx = data.edge_index.cpu().T
            G.add_edges_from(edge_idx.tolist())
        return G

    def add_faces(self) -> tnx.SimplicialComplex:
        """
        STEP 2: Add faces (triangles and Tetrahedrons) to the existing undirected graph G

        @param G: Fully configured NetworkX graph
        @type G: nx.Graph
        @return: TopoX Simplicial Complex
        @rtype: tnx.SimplicialComplex
        """
        # If faces have to be generated...
        if self.max_rank > 0:
            # If this is a smaller graph that we can extract the cliques directly

            if self.data.num_nodes < self.threshold_clique_complex:
                logging.info(f'Generate complex from cliques with {self.data.num_nodes} nodes')
                return graph_to_clique_complex(self.nx_graph, self.max_rank)
            else:
                logging.info(f'Generate complex from neighbors with {self.data.num_nodes} nodes')
                return graph_to_neighbor_complex(self.nx_graph)
        # Otherwise just generate the graph
        else:
            return tnx.SimplicialComplex(self.nx_graph)

    @staticmethod
    def features_from_hodge_laplacian(cplx: tnx.SimplicialComplex,
                                      num_eigenvectors: (int, int, int)) \
            -> (List[AbstractSimplicialComplex], List[AbstractSimplicialComplex], List[AbstractSimplicialComplex]):
        """
        STEP 3: Add features values to each node
        STEP 4: Add features to edges
        STEP 5: Add features to faces

        @param cplx: TopoX simplicial complex
        @type cplx: tnx.SimplicialComplex
        @param num_eigenvectors: Number of eigenvector to generate the features values
        @type num_eigenvectors: int
        @return: Fully configured elements of the simplicial complex
        @rtype: AbstractSimplicialComplex
        """
        from toponetx.algorithms.spectrum import hodge_laplacian_eigenvectors

        # Compute the laplacian weights for nodes, edges (L1) and faces (L2)
        simplicial_features = \
            [hodge_laplacian_eigenvectors(cplx.hodge_laplacian_matrix(idx),
                                          num_eigenvectors[idx])[1]
             for idx in range(len(num_eigenvectors))]
        # Generate the simplices related to node, edge and faces (triangles and tetrahedrons)
        return [GraphToSimplicial.__compute_simplicial_elements(cplx, simplicial_features, idx)
                for idx in range(len(simplicial_features))]

    """ ----------------  Private Helper Methods -------------------- """

    @staticmethod
    def __compute_simplicial_elements(cplx: tnx.SimplicialComplex,
                                      simplicial_features: List,
                                      index: int) -> List[SimplicialElement]:
        # Create simplicial element containing node indices associated with the simplex and feature set
        simplicial_node_feat = zip(cplx.skeleton(index), np.array(simplicial_features[index]), strict=True)
        return [SimplicialElement(tuple(u), v) for u, v in simplicial_node_feat]

