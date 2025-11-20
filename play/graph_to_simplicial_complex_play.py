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
from typing import AnyStr, Dict, Any, Callable, List, Tuple
import logging
# 3rd Party imports
import networkx as nx
import toponetx as tnx
# Library imports
from play import Play
from topology.simplicial import lift_from_graph_cliques, lift_from_graph_neighbors
from topology.simplicial.featured_simplex import FeaturedSimplex
from topology.simplicial.graph_to_simplicial_complex import GraphToSimplicialComplex
from deeplearning.training import TrainingException
import python


class GraphToSimplicialComplexPlay(Play):
    """
    Source code related to the Substack article 'Topological Lifting of Graph Neural Networks'

    References:
    - Substack: https://patricknicolas.substack.com/p/topological-lifting-of-graph-neural
    -  GraphToSimplicial:
        https://github.com/patnicolas/geometriclearning/blob/main/python/topology/simplicial/graph_to_simplicial_complex.py
        Github TopoNetX: https://github.com/pyt-team/TopoNetX/blob/main/toponetx/transform/graph_to_simplicial_complex.py

    The features are implemented by the class GraphToSimplicialComplex in the source file
                  python/topology/simplicial/graph_to_simplicial_complex.py
    The class GraphToSimplicialComplexPlay is a wrapper of the class GraphToSimplicialComplex
    The execution of the tests follows the same order as in the Substack article
    """
    def __init__(self,
                 dataset_name: AnyStr,
                 lifting_method: Callable[[nx.Graph, Dict[str, Any]], tnx.SimplicialComplex]) -> None:
        """
        Constructor for the evaluation of the topological lifting methods as described in 'Topological Lifting of
        Graph Neural Networks' Substack article

        @param dataset_name: Name of PyTorch Geometric dataset
        @type dataset_name: str
        @param lifting_method: Lifting method wrapping TopoNetX function
        @type lifting_method: Callable
        """
        super(GraphToSimplicialComplexPlay, self).__init__()

        self.dataset_name = dataset_name
        self.lifting_method = lifting_method

    def play(self) -> None:
        """
        Implementation of evaluation code as described in Substack article, including timing of execution
        """
        import time

        # Step 1: Configure the migration from Graph to Simplicial
        start = time.time()
        graph_to_simplicial = GraphToSimplicialComplex[AnyStr](dataset=self.dataset_name,
                                                               nx_graph=None,
                                                               lifting_method=self.lifting_method)
        # Step 2: Add faces to existing graph nodes and edges
        tnx_simplicial = graph_to_simplicial.add_faces({'max_rank': 2})

        # Step 3: Generate the simplicial elements for nodes, edges and faces.
        #         Number of eigenvectors for node is 4, edges 5 and faces 4
        num_eigenvectors = (4, 5, 4)
        featured_simplices = GraphToSimplicialComplex.features_from_hodge_laplacian(tnx_simplicial,
                                                                                    num_eigenvectors)

        self.__output_simplicial(featured_simplices, num_eigenvectors)
        logging.info(f'Duration: {time.time() - start}')

    def __output_simplicial(self,
                            featured_simplices: List[FeaturedSimplex],
                            num_eigenvectors: Tuple[int, int, int]) -> None:
        featured_nodes = [simplex for simplex in featured_simplices if simplex.get_rank() == 0]
        featured_edges = [simplex for simplex in featured_simplices if simplex.get_rank() == 1]
        featured_faces = [simplex for simplex in featured_simplices if simplex.get_rank() > 1]
        logging.info(
            f"{self.dataset_name}: {len(featured_nodes)} nodes,  {len(featured_edges)} edges, "
            f"{len(featured_faces)} faces")

        # Step 4: Prepare data for dumping results
        nodes_elements = [node for idx, node, in enumerate(featured_nodes) if idx < 3]
        edges_elements = [edge for idx, edge, in enumerate(featured_edges) if idx < 3]
        faces_elements = [face for idx, face, in enumerate(featured_faces) if idx < 3]
        nodes_elements_str = '\n'.join([str(s) for s in nodes_elements])
        edges_elements_str = '\n'.join([str(s) for s in edges_elements])
        faces_elements_str = '\n'.join([str(s) for s in faces_elements])
        logging.info(f"\nNodes: {num_eigenvectors[0]}, Edges: {num_eigenvectors[1]}, Faces: {num_eigenvectors[2]} "
                     f"eigenvectors\nSimplicial nodes:\n{nodes_elements_str}\nSimplicial edges:\n{edges_elements_str}"
                     f"\nSimplicial faces:\n{faces_elements_str}")


if __name__ == '__main__':
    try:
        # Test 1 - Code snippet 4
        topological_lifting_tutorial = GraphToSimplicialComplexPlay(dataset_name='Cora',
                                                                    lifting_method=lift_from_graph_cliques)
        topological_lifting_tutorial.play()
        topological_lifting_tutorial = GraphToSimplicialComplexPlay(dataset_name='KarateClub',
                                                                    lifting_method=lift_from_graph_neighbors)
        topological_lifting_tutorial.play()

        # Test 2 - Code snippet 6
        topological_lifting_tutorial = GraphToSimplicialComplexPlay(dataset_name='PubMed',
                                                                    lifting_method=lift_from_graph_cliques)
        topological_lifting_tutorial.play()
        topological_lifting_tutorial = GraphToSimplicialComplexPlay(dataset_name='Cora',
                                                                    lifting_method=lift_from_graph_neighbors)
        topological_lifting_tutorial.play()
    except AssertionError as e:
        logging.error(e)
        assert False
    except TrainingException as e:
        logging.error(e)
        assert False




