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
from typing import Dict, List, AnyStr, Generic, TypeVar, Callable, Any
# 3rd Party imports
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import toponetx as tnx
# Library imports
import python
from topology.simplicial.abstract_simplicial_complex import SimplicialElement, AbstractSimplicialComplex
_all_ = ['GraphToSimplicialComplex']

# Supporting types
T = TypeVar('T')
NXGraph = nx.Graph | None


class GraphToSimplicialComplex(Generic[T]):
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
                 nx_graph: NXGraph,
                 dataset: T,
                 lifting_method: Callable[[nx.Graph, Dict[str, Any]], tnx.SimplicialComplex]) -> None:
        """
        Constructor for the Generator of simplicial complex from a Graph.
        A NetworkX can be optionally provided. If not it will be initialized by method initialize_networkx_graph
        A PyTorch Geometric sataset is provided either as a name or the dataset itself

        @param nx_graph: Optional NetworkX graph.
        @type nx_graph: Union[nx.Graph, None]
        @param dataset: Name of PyTorch Geometric data set OR dataset itself
        @type dataset: Union[str, Dataset]
        @param lifting_method: Lifting method from a graph to a simplicial complex using various methods
                               Extraction of NetworkX graph cliques
                               Collection of node neighbors
                               Generation of Vietoris-Rips complext
        @type lifting_method: Callable[[nx.Graph], tnx.SimplicialComplex]
        """
        # If the dataset is provided using its name
        if type(dataset).__name__ == 'str':
            from dataset.graph.pyg_datasets import PyGDatasets

            # The class PyGDatasets validate the dataset is part of PyTorch Geometric Library
            pyg_dataset = PyGDatasets(dataset)
            dataset = pyg_dataset()
            self.dataset_name = dataset
        # If the dataset is provided as part of PyTorch Geometric Library
        elif str(type(dataset).__module__) == 'torch_geometric':
            self.dataset_name = getattr(dataset, 'name')
        else:
            raise TypeError(f'Dataset has incorrect type {str(type(dataset))}')

        # Initialize the NetworkX graph if not provided in the constructor
        self.nx_graph = GraphToSimplicialComplex.__build_networkx_graph(nx_graph, dataset[0])
        self.data = dataset[0]
        self.lifting_method = lifting_method

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
        return {GraphToSimplicialComplex.types_map[length]: sum(1 for _ in group)
                for length, group in groupby(sorted_simplex, key=len) if length < 5}

    @staticmethod
    def __build_networkx_graph(G: nx.Graph, data: Data) -> nx.Graph:
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

    def add_faces(self, params: Dict[str, Any] = None) -> tnx.SimplicialComplex:
        """
        STEP 2: Add faces (triangles and Tetrahedrons) to the existing undirected graph G

        @return: TopoX Simplicial Complex
        @rtype: tnx.SimplicialComplex
        """
        return self.lifting_method(self.nx_graph, params)

    @staticmethod
    def features_from_hodge_laplacian(tnx_simplicial: tnx.SimplicialComplex,
                                      num_eigenvectors: (int, int, int)) \
            -> (List[SimplicialElement], List[SimplicialElement], List[SimplicialElement]):
        """
        STEP 3: Add features values to each node
        STEP 4: Add features to edges
        STEP 5: Add features to faces

        @param tnx_simplicial: TopoX simplicial complex
        @type tnx_simplicial: tnx.SimplicialComplex
        @param num_eigenvectors: Number of eigenvector to generate the features values
        @type num_eigenvectors: int
        @return: Fully configured elements of the simplicial complex
        @rtype: AbstractSimplicialComplex
        """
        from toponetx.algorithms.spectrum import hodge_laplacian_eigenvectors

        # Compute the laplacian weights for nodes, edges (L1) and faces (L2)
        simplicial_features = \
            [hodge_laplacian_eigenvectors(tnx_simplicial.hodge_laplacian_matrix(idx),
                                          num_eigenvectors[idx])[1]
             for idx in range(len(num_eigenvectors))]
        # Generate the simplices related to node, edge and faces (triangles and tetrahedrons)
        return [GraphToSimplicialComplex.__compute_simplicial_elements(tnx_simplicial, simplicial_features, idx)
                for idx in range(len(simplicial_features))]

    """ ----------------  Private Helper Methods -------------------- """

    @staticmethod
    def __compute_simplicial_elements(tnx_simplicial: tnx.SimplicialComplex,
                                      simplicial_features: List,
                                      index: int) -> List[SimplicialElement]:
        # Create simplicial element containing node indices associated with the simplex and feature set
        simplicial_node_feat = zip(tnx_simplicial.skeleton(index), np.array(simplicial_features[index]), strict=True)
        return [SimplicialElement(tuple(u), v) for u, v in simplicial_node_feat]

