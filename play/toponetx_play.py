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
from typing import AnyStr, Callable, Dict, Any, List, Generic, TypeVar, Tuple
import logging
from toponetx import CellComplex, SimplicialComplex
# 3rd Party imports
from toponetx.classes.complex import Complex
from torch.utils.data import Dataset
import toponetx as tnx
import networkx as nx
import numpy as np
# Library imports
from play import Play
from topology import TopologyException
from topology.simplicial.abstract_simplicial_complex import ComplexElement
from play.graph_to_simplicial_complex_play import lift_from_graph_cliques

# Supporting types
T = TypeVar('T')


class TopoNetXPlay(Play, Generic[T]):
    def __init__(self, dataset: T, this_complex: Complex = None) -> None:
        super(Play, self).__init__()
        self.dataset, self.dataset_name = TopoNetXPlay.__extract_dataset_name(dataset)
        self.complex = this_complex

        if this_complex is not None:
            if not isinstance(this_complex, (SimplicialComplex, CellComplex)):
                raise TypeError(f"Unsupported topological complex type: {type(this_complex).__name__}")

    def lift(self,
             lifting_method: Callable[[nx.Graph, Dict[str, Any]], tnx.Complex],
             params: Dict[AnyStr, Any],
             num_eigenvectors: Tuple[int, int, int]) -> None:
        if self.complex is not None:
            raise TopologyException(f'The complex of type  {type(self.complex).__name__} is already defined')

        nx_graph = self.__build_networkx_graph()
        if self.complex is None:
            self.complex = lifting_method(nx_graph, params)

            node_simplicial_elements, edge_simplicial_elements, face_simplicial_elements = (
                self.simplex_features_from_hodge_laplacian(num_eigenvectors)
            )
            self.__output_simplicial(node_simplicial_elements,
                                     edge_simplicial_elements,
                                     face_simplicial_elements,
                                     num_eigenvectors)

    def simplex_features_from_hodge_laplacian(self, num_eigenvectors: (int, int, int)) \
            -> (List[ComplexElement], List[ComplexElement], List[ComplexElement]):
        """
        STEP 3: Add features values to each node
        STEP 4: Add features to edges
        STEP 5: Add features to faces

        @param num_eigenvectors: Number of eigenvector to generate the features values
        @type num_eigenvectors: int
        @return: Fully configured elements of the simplicial complex
        @rtype: AbstractSimplicialComplex
        """
        from toponetx.algorithms.spectrum import hodge_laplacian_eigenvectors

        # Compute the laplacian weights for nodes, edges (L1) and faces (L2)
        complex_features = \
            [hodge_laplacian_eigenvectors(self.complex.hodge_laplacian_matrix(idx),
                                          num_eigenvectors[idx])[1]
             for idx in range(len(num_eigenvectors))]
        # Generate the simplices related to node, edge and faces (triangles and tetrahedrons)
        return [TopoNetXPlay.__compute_complex_elements(complex_features, idx)
                for idx in range(len(complex_features))]

    """ ----------------------  Private Helper Methods ---------------------- """

    def __output_simplicial(self,
                            node_elements: List[ComplexElement],
                            edge_elements: List[ComplexElement],
                            face_elements: List[ComplexElement],
                            num_eigenvectors: Tuple[int, int, int]
                            ) -> None:
        logging.info(
            f"{self.dataset_name}: {len(node_elements)} nodes,  {len(edge_elements)} edges, "
            f"{len(face_elements)} faces")

        # Step 4: Prepare data for dumping results
        nodes_elements = [node for idx, node, in enumerate(node_elements) if idx < 3]
        edges_elements = [edge for idx, edge, in enumerate(edge_elements) if idx < 3]
        faces_elements = [face for idx, face, in enumerate(face_elements) if idx < 3]
        nodes_elements_str = '\n'.join([str(s) for s in nodes_elements])
        edges_elements_str = '\n'.join([str(s) for s in edges_elements])
        faces_elements_str = '\n'.join([str(s) for s in faces_elements])
        logging.info(f"\nNodes: {num_eigenvectors[0]}, Edges: {num_eigenvectors[1]}, Faces: {num_eigenvectors[2]} "
                     f"eigenvectors\nSimplicial nodes:\n{nodes_elements_str}\nSimplicial edges:\n{edges_elements_str}"
                     f"\nSimplicial faces:\n{faces_elements_str}")

    @staticmethod
    def __extract_dataset_name(in_dataset: T) -> (Dataset, AnyStr):
        # If the dataset is provided using its name
        if type(in_dataset).__name__ == 'str':
            from dataset.graph.pyg_datasets import PyGDatasets

            # The class PyGDatasets validate the dataset is part of PyTorch Geometric Library
            pyg_dataset = PyGDatasets(in_dataset)
            dataset = pyg_dataset()
            return (dataset, in_dataset)

        # If the dataset is provided as part of PyTorch Geometric Library
        elif str(type(in_dataset).__module__) == 'torch_geometric':
            return (in_dataset, getattr(in_dataset, 'name'))
        else:
            raise TypeError(f'Dataset has incorrect type {str(type(in_dataset))}')

    def __compute_complex_elements(self,
                                   simplicial_features: List,
                                   index: int) -> List[ComplexElement]:
        # Create simplicial element containing node indices associated with the simplex and feature set
        simplicial_node_feat = zip(self.complex.skeleton(index), np.array(simplicial_features[index]), strict=True)
        return [ComplexElement(tuple(u), v) for u, v in simplicial_node_feat]

    def __build_networkx_graph(self) -> nx.Graph:
        """
        STEP 1: Initialization of the graph if it has not been initialized.

        @return: NetworkX undirected graph
        @rtype: NetworkX Graph
        """
        data = self.dataset[0]
        # Create a NetworkX graph
        G = nx.Graph()

        # Populate with the node from the dataset
        G.add_nodes_from(range(data.num_nodes))

        # Populate with the edges from the dataset: We need to transpose the tensor from 2 x num edges shape to
        # num edges x 2 shape
        edge_idx = data.edge_index.cpu().T
        G.add_edges_from(edge_idx.tolist())
        return G



if __name__ == '__main__':
    toponetx_play = TopoNetXPlay[SimplicialComplex](dataset='Cora', this_complex=None)
    toponetx_play.lift(lifting_method=lift_from_graph_cliques, params={'max_rank': 2}, num_eigenvectors=(4, 5, 4))






