__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

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
import logging
import python
# 3rd party library imports
import numpy as np
# Library imports
from play import Play
from topology.hypergraph.featured_hypergraph import FeaturedHyperGraph
from topology.complex_laplacian import ComplexLaplacian
from topology import LaplacianType


class FeaturedHyperGraphPlay(Play):
    """
    Wrapper to implement the evaluation of hypergraphs as defined in Substack article:
    "Exploring Hypergraphs with TopoX Library"

    References:
    - Article: https://patricknicolas.substack.com/p/exploring-hypergraphs-with-topox
    - Implementation
      https://github.com/patnicolas/geometriclearning/blob/main/python/topology/hypergraph/featured_hyperedge.py
      https://github.com/patnicolas/geometriclearning/blob/main/python/topology/hypergraph/featured_hypergraph.py
    - Evaluation
      https://github.com/patnicolas/geometriclearning/blob/main/play/featured_hypergraph_play.py

    The features are implemented by the class FeaturedHypergraph in the source file
                       python/topology/hypergraph/featured_hypergraph.py
    The class FeaturedHypergraphPlay is a wrapper of the class FeaturedHypergraph
    The execution of the tests follows the same order as in the Substack article
    """
    def __init__(self) -> None:
        super(FeaturedHyperGraphPlay, self).__init__()
        self.featured_hypergraph = FeaturedHyperGraphPlay.__generate_featured_hypergraph()
        logging.info(self.featured_hypergraph)

    def play(self) -> None:
        """
        Sequence of tests used in the substack article "Exploring Hypergraphs with TopoX Library"
        https://patricknicolas.substack.com/p/exploring-hypergraphs-with-topox
        """
        self.play_adjacency()
        self.play_incidence()
        self.play_conversion_simplicial_complex()
        self.play_laplacians()

    def play_adjacency(self) -> None:
        """
         Implementation of the evaluation code for the Substack article "Exploring Hypergraphs with TopoX Library"
         Code snippet 4
        """
        adjacency_matrix = self.featured_hypergraph.adjacency_matrix(ranks=(0, 1))
        logging.info(f'\nAdjacency Matrix (0, 1):\n{adjacency_matrix}')

        adjacency_matrix = self.featured_hypergraph.adjacency_matrix(ranks=(0, 2))
        logging.info(f'\nAdjacency Matrix (0, 2):\n{adjacency_matrix}')

        adjacency_matrix = self.featured_hypergraph.adjacency_matrix(ranks=(1, 2))
        logging.info(f'\nAdjacency Matrix (1, 2):\n{adjacency_matrix}')

        adjacency_matrix = self.featured_hypergraph.adjacency_matrix(ranks=(1, 3))
        logging.info(f'\nAdjacency Matrix (1, 3):\n{adjacency_matrix}')

    def play_incidence(self) -> None:
        """
        Implementation of the evaluation code for the Substack article "Exploring Hypergraphs with TopoX Library"
        Code snippet 4
        """
        incidence_matrix = self.featured_hypergraph.incidence_matrix(ranks=(0, 1))
        logging.info(f'\nIncidence Matrix (0, 1):\n{incidence_matrix}')

        incidence_matrix = self.featured_hypergraph.adjacency_matrix(ranks=(0, 2))
        logging.info(f'\nincidence Matrix (0, 2):\n{incidence_matrix}')

        incidence_matrix = self.featured_hypergraph.adjacency_matrix(ranks=(1, 2))
        logging.info(f'\nincidence Matrix (1, 2):\n{incidence_matrix}')

    def play_conversion_simplicial_complex(self) -> None:
        """
        Implementation of the evaluation code for the Substack article "Exploring Hypergraphs with TopoX Library"
        Code snippet 6
        """
        self.featured_hypergraph.set_simplicial_complex()
        logging.info(f'Extracted simplicial complex indices:\n{self.featured_hypergraph.simplicial_complex_dict}')

    def play_laplacians(self) -> None:
        """
        Implementation of the evaluation code for the Substack article "Exploring Hypergraphs with TopoX Library"
        Code snippet 6
        """
        self.featured_hypergraph.set_simplicial_complex()
        test_rank = 1
        complex_laplacian = ComplexLaplacian(
            laplacian_type=LaplacianType.HodgeLaplacian,
            rank=test_rank,
            signed=False
        )
        laplacian = self.featured_hypergraph.laplacian(complex_laplacian)
        logging.info(f'\nHodge Laplacian rank {test_rank}:\n{laplacian}')

        test_rank = 2
        complex_laplacian = ComplexLaplacian(
            laplacian_type=LaplacianType.HodgeLaplacian,
            rank=test_rank,
            signed=False
        )
        laplacian = self.featured_hypergraph.laplacian(complex_laplacian)
        logging.info(f'\nHodge Laplacian rank {test_rank}:\n{laplacian}')

    @staticmethod
    def __generate_featured_hypergraph() -> FeaturedHyperGraph:
        hyperedge_indices_list = frozenset([(1, 2), (1, 4), (1, 2, 4, 3), (3, 5), (5, 6, 4), (1, 2, 4, 3, 5, 6)])
        rank_list = [1, 1, 2, 1, 2, 3]
        features_list = [
            np.array([0.5, 0.8]),
            np.array([0.1, 0.0]),
            np.array([0.0, 0.5]),
            np.array([0.9, 0.1]),
            np.array([0.2, 0.5]),
             np.array([1.0, 0.2])
        ]
        return FeaturedHyperGraph.build(
            hyperedge_indices_list=hyperedge_indices_list,
            ranks=rank_list,
            features_list=features_list
        )


if __name__ == '__main__':
    try:
        featured_hypergraph_play = FeaturedHyperGraphPlay()
        featured_hypergraph_play.play()
    except (ValueError, TypeError, TypeError) as e:
        logging.error(e)
