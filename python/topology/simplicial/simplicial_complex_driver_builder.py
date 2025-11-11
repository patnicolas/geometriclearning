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
from typing import AnyStr, Dict, Callable, Any, Tuple
# 3rd Party imports
from torch_geometric.data import Dataset
import networkx as nx
import toponetx as tnx
# Library imports
from topology.hodge_spectrum_configuration import HodgeSpectrumConfiguration
from topology.simplicial.graph_to_simplicial_complex import GraphToSimplicialComplex
from topology.simplicial.featured_simplicial_complex import FeaturedSimplicialComplex
from topology.networkx_graph import NetworkxGraph


class SimplicialComplexDriverBuilder(object):
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

        # If Graph data structure is not provided
        if nx.graph is None:
            networkx_graph = NetworkxGraph(dataset[0])
            self.nx_graph = networkx_graph.G
        self.nx_graph = nx_graph
        self.graph_complex_elements = None

    def __call__(self,
                 num_eigenvectors: Tuple[int, int, int],
                 lifting_method: Callable[[nx.Graph, Dict[str, Any]], tnx.SimplicialComplex])\
            -> FeaturedSimplicialComplex:
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
            @param lifting_method:  Lifting method from graph to a Simplicial Complex
            @type lifting_method: Callable
            @return: New Simplicial elements
            @rtype: FeaturedSimplicialComplex
        """
        graph_to_simplicial = GraphToSimplicialComplex(self.nx_graph, self.dataset, lifting_method)
        tnx_complex = graph_to_simplicial.add_faces()

        hodge_spectrum_config = HodgeSpectrumConfiguration(num_eigenvectors)
        graph_complex_elements = hodge_spectrum_config.get_complex_features(tnx_complex)

        return FeaturedSimplicialComplex(graph_complex_elements)
