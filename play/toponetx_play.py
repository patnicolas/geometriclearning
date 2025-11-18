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


# Python standard library imports
from typing import AnyStr, Callable, Dict, Any, List, Generic, TypeVar, Tuple, Self
import logging
# 3rd Party imports
from toponetx.classes.complex import Complex
from toponetx import CellComplex, SimplicialComplex
from toponetx.generators.random_cell_complexes import np_cell_complex
from toponetx.generators.random_simplicial_complexes import random_clique_complex
from torch.utils.data import Dataset
import toponetx as tnx
import networkx as nx
# Library imports
from play import Play
from play.graph_to_simplicial_complex_play import lift_from_graph_cliques
from topology.hodge_spectrum_configuration import HodgeSpectrumConfiguration

# Supporting types
T = TypeVar('T')


class TopoNetXPlay(Play, Generic[T]):
    """
    This class wraps the evaluation of the TopoNetX library for Simplicial and Cell complex. There are 3 constructor
    for the class:
       __init__: default with data set and predefined simplicial and cell complex
       build_from_lift:  Complex generated through a partial lift from an existing graph data
       build_from_random: Random generated complex using TopoNetX generator module

    Simplicial Complexes ------
    A simplicial complex is a particular kind of topological space built by joining together simplices—points,
    line segments, triangles, and their higher-dimensional analogs.  It generalizes the idea of a triangle in a
    triangulated surface or a tetrahedron in a three-dimensional manifold decomposition.

    Simplicial complexes are the foundational structures in combinatorial topology, providing a discrete way to
    analyze continuous spaces. For instance, a triangle can be viewed as a simplicial complex composed of three
    vertices connected pairwise by edges, forming a 2-simplex. Similarly, a tetrahedron is a simplicial complex
    consisting of four vertices, six edges, and four triangular faces, representing a 3-simplex.

    Cell Complexes --------
    A cell complex is a mathematical framework constructed from elementary building blocks known as cells. These cells
    generalize familiar geometric shapes—such as points, line segments, triangles, and disks. By systematically
    “gluing” these cells together according to specific rules, one can form intricate geometric structures that serve
    as fundamental objects of study in topology and geometry.

    Cell complexes provide a flexible way to represent diverse mathematical entities, including graphs, manifolds,
    and other discrete geometric forms. They play a central role in algebraic topology and geometric analysis, where
    they are used to investigate the shape, connectivity, and higher-order properties of these objects.
    """
    def __init__(self, dataset: T, this_complex: Complex) -> None:
        """
        Default constructor for the evaluation of the TopoNetX library for which a fully defined simplicial or
        cell complex is provided.
        A TypeError is thrown if complex type is not supported

        @param dataset: Pytorch Geometric dataset used for evaluation
        @type dataset: Generic (Str or Dataset type)
        @param this_complex: Simplicial or Cell complex
        @type this_complex: Complex
        """
        super(Play, self).__init__()

        # TypeError is thrown if complex type is not supported
        if not isinstance(this_complex, (SimplicialComplex, CellComplex)):
            raise TypeError(f"Unsupported topological complex type: {type(this_complex).__name__}")

        self.dataset, self.dataset_name = TopoNetXPlay.__extract_dataset_name(dataset)
        self.complex = this_complex

    @classmethod
    def build_from_lift(cls,
                        dataset: T,
                        lifting_method: Callable[[nx.Graph, Dict[str, Any]], tnx.Complex],
                        params: Dict[AnyStr, Any]) -> Self:
        """
        Alternative constructor that relies on a lifting method and predefined configuration parameters.

        Lifting a graph G=(V, E) into a complex Cplx=(V, E, C) means augmenting it with higher-dimensional entities
        (faces, volumes, etc.) so that you can capture relationships beyond pairwise edges.
        This implementation is a 'partial' lift. Full Simplicial lifting assigns signal/features to faces or triangles.
        Partial Simplicial lifting uses topological domains in the message passing and aggregation only
        (e.g., L1 Hodge Laplacian).

        @see Substack article on Topological Lifting of Graph Neural Networks
            https://patricknicolas.substack.com/p/topological-lifting-of-graph-neural


        @param dataset: PyTorch Geometric graph dataset
        @type dataset: torch.utils.data.Dataset or str
        @param lifting_method: Lifting method to be used in generating the complex
        @type lifting_method: Callable (Lambda)
        @param params: Configuration parameters such as max_rank, weighs used in lift
        @type params: Dictionary
        @return: Instance of this class
        @rtype: TopoNetXPlay
        """
        # Initialize the graph
        nx_graph = TopoNetXPlay.__build_networkx_graph(dataset)
        # Apply the appropriate lifting method to generate the complex
        this_complex = lifting_method(nx_graph, params)
        return cls(dataset, this_complex)

    @classmethod
    def build_from_random(cls,
                          dataset: T,
                          complex_type: AnyStr,
                          num_nodes: int,
                          prob_edge: float) -> Self:
        """
        Alternative constructor to generate an instance for the evaluation of TopoNetX library

        @param dataset: PyTorch Geometric graph dataset
        @type dataset: torch.utils.data.Dataset or str
        @param complex_type: Type of complex
        @type complex_type: str
        @param num_nodes: Number of nodes for random generation of the complex
        @type num_nodes: int
        @param prob_edge: Probability of an edge exist between any given nodes
        @type prob_edge: float
        @return: Instance of this class
        @rtype: TopoNetXPlay
        """
        match complex_type:
            case 'simplicial':
                return cls(dataset, random_clique_complex(n=num_nodes, p=prob_edge))
            case 'cell':
                return cls(dataset, np_cell_complex(n=num_nodes, p=prob_edge))
            case _:
                raise TypeError(f'{complex_type} complex is not supported')

    def __str__(self) -> AnyStr:
        return f'\nDataset: {self.dataset}\n{str(self.complex)}'

    """ ----------------------  Private Helper Methods ---------------------- """

    @staticmethod
    def __extract_dataset_name(in_dataset: T) -> (Dataset, AnyStr):
        # If the dataset is provided using its name
        if type(in_dataset).__name__ == 'str':
            from dataset.graph.pyg_datasets import PyGDatasets

            # The class PyGDatasets validate the dataset is part of PyTorch Geometric Library
            pyg_dataset = PyGDatasets(in_dataset)
            dataset = pyg_dataset()
            return dataset, in_dataset

        # If the dataset is provided as part of PyTorch Geometric Library
        elif str(type(in_dataset).__module__) == 'torch_geometric':
            return in_dataset, getattr(in_dataset, 'name')
        else:
            raise TypeError(f'Dataset has incorrect type {str(type(in_dataset))}')

    @staticmethod
    def __build_networkx_graph(in_dataset: Dataset) -> nx.Graph:
        """
        STEP 1: Initialization of the graph if it has not been initialized.

        @return: NetworkX undirected graph
        @rtype: NetworkX Graph
        """
        if type(in_dataset).__name__ == 'str':
            from dataset.graph.pyg_datasets import PyGDatasets

            # The class PyGDatasets validate the dataset is part of PyTorch Geometric Library
            pyg_dataset = PyGDatasets(in_dataset)
            dataset = pyg_dataset()
        else:
            dataset = in_dataset

        data = dataset[0]
        # Create a NetworkX graph
        G = nx.Graph()

        # Populate with the node from the dataset
        G.add_nodes_from(range(data.num_nodes))

        # Populate with the edges from the dataset: We need to transpose the tensor from 2 x num edges shape to
        # num edges x 2 shape
        edge_idx = data.edge_index.cpu().CellDescriptor
        G.add_edges_from(edge_idx.tolist())
        return G


def play_build_simplicial_complex(dataset_name: AnyStr) -> None:
    logging.info('\nSimplicial from Lift')
    toponetx_play = TopoNetXPlay[SimplicialComplex].build_from_lift(dataset=dataset_name,
                                                                    lifting_method=lift_from_graph_cliques,
                                                                    params={'max_rank': 2})
    logging.info(toponetx_play)
    hodge_spectrum_config = HodgeSpectrumConfiguration.build(num_node_eigenvectors=4,
                                                             num_edge_eigenvectors=5,
                                                             num_simplex_2_eigenvectors=4)

    lifted_complex_elements = hodge_spectrum_config.get_complex_features(toponetx_play.complex)
    logging.info(f'\n{lifted_complex_elements.dump(4)}')
    logging.info('\nSimplicial from random')
    toponetx_play = TopoNetXPlay[SimplicialComplex].build_from_random(dataset=dataset_name,
                                                                      complex_type='simplicial',
                                                                      num_nodes=64,
                                                                      prob_edge=0.44)
    logging.info(toponetx_play)
    lifted_complex_elements = hodge_spectrum_config.get_complex_features(toponetx_play.complex)
    logging.info(f'\n{lifted_complex_elements.dump(4)}')


def play_build_cell_complex(dataset_name: AnyStr) -> None:
    logging.info('\nCell complex from Lift')
    toponetx_play = TopoNetXPlay[CellComplex].build_from_lift(dataset=dataset_name,
                                                              lifting_method=lift_from_graph_cliques,
                                                              params={'max_rank': 2})
    logging.info(toponetx_play)
    hodge_spectrum_config = HodgeSpectrumConfiguration.build(num_node_eigenvectors=4,
                                                             num_edge_eigenvectors=5,
                                                             num_simplex_2_eigenvectors=4)

    lifted_complex_elements = hodge_spectrum_config.get_complex_features(toponetx_play.complex)
    logging.info(f'\n{lifted_complex_elements.dump(4)}')
    logging.info('\nSimplicial from random')
    toponetx_play = TopoNetXPlay[CellComplex].build_from_random(dataset=dataset_name,
                                                                complex_type='cell',
                                                                num_nodes=64,
                                                                prob_edge=0.44)
    logging.info(toponetx_play)
    lifted_complex_elements = hodge_spectrum_config.get_complex_features(toponetx_play.complex)
    logging.info(f'\n{lifted_complex_elements.dump(4)}')


if __name__ == '__main__':
    # play_build_simplicial_complex('Cora')
    play_build_cell_complex('Cora')
    #play_build_simplicial_complex('PubMed')
    play_build_cell_complex('PubMed')