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
from typing import List, Tuple
import logging
# Library imports
from play import Play
from topology.simplicial.featured_simplicial_complex import FeaturedSimplicialComplex
from topology.complex_laplacian import LaplacianType, ComplexLaplacian


class AbstractSimplicialComplexPlay(Play):
    """
    Wrapper to implement the evaluation of Simplicial Simplex as defined in Substack article:
    "Exploring Simplicial Complexes for Deep Learning: Concepts to Code"

    References:
    - Article: https://patricknicolas.substack.com/p/exploring-simplicial-complexes-for
    - AbstractSimplicialComplex:
        https://github.com/patnicolas/geometriclearning/blob/main/python/topology/simplicial/abstract_simplicial_complex.py


  The features are implemented by the class AbstractSimplicialComplex in the source file
                  python/topology/simplicial/featured_simplicial_complex.py
    The class AbstractSimplicialComplexPlay is a wrapper of the class AbstractSimplicialComplex
    The execution of the tests follows the same order as in the Substack article
    """
    def __init__(self,
                 node_feature_dimension: int,
                 edge_set: List[Tuple[int, ...]],
                 face_set: List[Tuple[int, ...]]) -> None:
        """
        Constructor for this tutorial
        @param node_feature_dimension: Dimension (number of features) for nodes
        @type node_feature_dimension: int
        @param edge_set: List of tuple (source_node_index, target_node_index)
        @type edge_set: List[Tuple[int, int]
        @param face_set: List of tuple for face/triangles (first_node_index, second_node_index, third_node_index)
        @type face_set: List[Tuple[int, int, int]
        """
        super(AbstractSimplicialComplexPlay, self).__init__()

        self.abstract_simplicial_complex = FeaturedSimplicialComplex.random(
            node_feature_dimension=node_feature_dimension,
            edge_node_indices=edge_set,
            face_node_indices=face_set)

    def play(self) -> None:
        """
            Implementation of the evaluation code for the Substack article "Exploring Simplicial Complexes for Deep
            Learning: Concepts to Code" - Code snippets 5, 7 & 9
        """
        self.play_adjacency()
        self.play_incidence()
        self.play_up_laplacian()
        self.play_down_laplacian()
        self.play_hodge_laplacian()

    # Test 1 - Code snippet 5
    def play_adjacency(self) -> None:
        logging.info(f'\nAdjacency matrix:\n{self.abstract_simplicial_complex.adjacency_matrix()}')

    # Test 2 - Code snippet 7
    def play_incidence(self) -> None:
        for rank in range(1, 3):
            incidence_matrix = self.abstract_simplicial_complex.incidence_matrix(rank=rank)
            logging.info(f'\nDirected incidence matrix rank {rank}:\n{incidence_matrix}')

    # Test 3 - Code snippet 9
    def play_up_laplacian(self) -> None:
        simplicial_laplacian_0 = ComplexLaplacian(laplacian_type=LaplacianType.UpLaplacian,
                                                  rank=0,
                                                  signed=True)
        up_laplacian_rk0 = self.abstract_simplicial_complex.laplacian(simplicial_laplacian_0)
        logging.info(f'\nUP-Laplacian rank 0\n{up_laplacian_rk0}')
        simplicial_laplacian_1 = ComplexLaplacian(laplacian_type=LaplacianType.UpLaplacian,
                                                  rank=1,
                                                  signed=True)
        up_laplacian_rk1 = self.abstract_simplicial_complex.laplacian(simplicial_laplacian_1)
        logging.info(f'\nUP-Laplacian rank 1\n{up_laplacian_rk1}')

    # Test 4
    def play_down_laplacian(self) -> None:
        simplicial_laplacian_1 = ComplexLaplacian(
            laplacian_type=LaplacianType.DownLaplacian,
            rank=1,
            signed=True)
        down_laplacian_rk1 = self.abstract_simplicial_complex.laplacian(simplicial_laplacian_1)
        logging.info(f'\nDown-Laplacian rank 1\n{down_laplacian_rk1}')
        simplicial_laplacian_2 = ComplexLaplacian(
            laplacian_type=LaplacianType.DownLaplacian,
            rank=2,
            signed=True)
        down_laplacian_rk2 = self.abstract_simplicial_complex.laplacian(simplicial_laplacian_2)
        logging.info(f'\nDown-Laplacian rank 2\n{down_laplacian_rk2}')

    # Test 5
    def play_hodge_laplacian(self) -> None:
        simplicial_laplacian_0 = ComplexLaplacian(
            laplacian_type=LaplacianType.HodgeLaplacian,
            rank=0,
            signed=True)
        hodge_laplacian_rk0 = self.abstract_simplicial_complex.laplacian(simplicial_laplacian_0)
        logging.info(f'\nHodge-Laplacian rank 0\n{hodge_laplacian_rk0}')

        simplicial_laplacian_1 = ComplexLaplacian(
            laplacian_type=LaplacianType.HodgeLaplacian,
            rank=1,
            signed=True)
        hodge_laplacian_rk1 = self.abstract_simplicial_complex.laplacian(simplicial_laplacian_1)
        logging.info(f'\nHodge-Laplacian rank 1\n{hodge_laplacian_rk1}')

        simplicial_laplacian_2 = ComplexLaplacian(
            laplacian_type=LaplacianType.HodgeLaplacian,
            rank=2,
            signed=True
        )
        hodge_laplacian_rk2 = self.abstract_simplicial_complex.laplacian(simplicial_laplacian_2)
        logging.info(f'\nHodge-Laplacian rank 2\n{hodge_laplacian_rk2}')


if __name__ == '__main__':
    test_edge_set = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5), (4, 5)]
    test_face_set = [(2, 3, 4), (1, 2, 3)]

    try:
        # Test 1
        simplicial_complex_play = AbstractSimplicialComplexPlay(node_feature_dimension=4,
                                                                edge_set=test_edge_set,
                                                                face_set=test_face_set)
        simplicial_complex_play.play_adjacency()

        # Test 2
        simplicial_complex_play_2 = AbstractSimplicialComplexPlay(node_feature_dimension=5,
                                                                  edge_set=test_edge_set,
                                                                  face_set=test_face_set)
        simplicial_complex_play_2.play_incidence()

        # Test 3
        simplicial_complex_play.play_up_laplacian()

        # Test 4
        simplicial_complex_play.play_down_laplacian()

        # Test 5
        simplicial_complex_play.play_hodge_laplacian()
        assert True
    except AssertionError as e:
        logging.error(e)
        assert False
