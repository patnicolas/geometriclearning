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
from typing import AnyStr, List, Tuple, Self
from dataclasses import dataclass
# 3rd Party imports
from toponetx.classes.complex import Complex
import numpy as np
# Library imports
from topology.simplicial.featured_simplex import FeaturedSimplex
# from topology.simplicial.featured_simplicial_elements import FeaturedSimplicialElements
__all__ = ['HodgeSpectrumConfiguration']


@dataclass
class HodgeSpectrumConfiguration:
    """
    Class to generate a set of complex elements from a graph using the Hodge Laplacian

    @param num_eigenvectors: Number of eigenvectors or eigenvalues used for each of the Indidence matrices for
                    nodes, edges and simplex-2
    @type num_eigenvectors: int
    """
    num_eigenvectors: Tuple[int, int, int]

    def __post_init__(self) -> None:
        """
        Post initialization validation of input values - Throw a ValueError exception is input are out of bounds
        """
        if self.num_eigenvectors[0] < 1:
            raise ValueError(f'\nNum of eigenvalues for nodes incidence {self.num_eigenvectors[0]} should be > 0')
        if self.num_eigenvectors[1] < 1:
            raise ValueError(f'\nNum of eigenvalues for edges incidence {self.num_eigenvectors[1]} should be > 0')
        if self.num_eigenvectors[2] < 1:
            raise ValueError(f'\nNum of eigenvalues for 2-simplex incidence {self.num_eigenvectors[2]} should be > 0')

    @classmethod
    def build(cls, num_node_eigenvectors: int, num_edge_eigenvectors: int, num_simplex_2_eigenvectors: int) -> Self:
        """
        Alternative constructor for defining the number of Eigenvectors for Hodge Laplacian. A Value error is thrown
        if number of eigenvalue is out-of-range

        @param num_node_eigenvectors:  Number of eigen vectors for incidence matrix for nodes
        @type num_node_eigenvectors: int
        @param num_edge_eigenvectors: Number of eigen vectors for incidence matrix for edges
        @type num_edge_eigenvectors: int
        @param num_simplex_2_eigenvectors: Number of eigen vectors for incidence matrix for simplex-2
        @type num_simplex_2_eigenvectors: int
        @return: Instance of HodgeLaplacianEigenvectors
        @rtype: HodgeSpectrumConfiguration
        """
        return cls((num_node_eigenvectors, num_edge_eigenvectors, num_simplex_2_eigenvectors))

    def get_num_node_eigenvalues(self) -> int:
        return self.num_eigenvectors[0]

    def get_num_edge_eigenvalues(self) -> int:
        return self.num_eigenvectors[1]

    def get_num_simplex_2_eigenvalues(self) -> int:
        return self.num_eigenvectors[2]

    def __str__(self) -> AnyStr:
        return (f'\n{self.num_eigenvectors[0]} node eigenvalues {self.num_eigenvectors[1]} edges eigenvalues '
                f'{self.num_eigenvectors[2]} edges eigenvalues')

    def get_complex_features(self, this_complex: Complex) -> List[FeaturedSimplex]:
        """
        Extract the simplex features for nodes, edges and simplex_2

        Steps
        1: Add features values to each node
        2: Add features to edges
        3: Add features to faces

        @param this_complex: This simplicial or cell complex
        @type this_complex: Complex
        @return: Fully configured, lifted elements of the complex
        @rtype: FeaturedSimplicialElements
        """
        from toponetx.algorithms.spectrum import hodge_laplacian_eigenvectors

        # Compute the laplacian weights for nodes, edges (L1) and faces (L2)
        complex_features = \
            [hodge_laplacian_eigenvectors(this_complex.hodge_laplacian_matrix(idx), self.num_eigenvectors[idx])[1]
             for idx in range(len(self.num_eigenvectors))]

        # Generate the simplex related to node, edge and simplex_2 (triangles, cells ...)
        complex_elements = [HodgeSpectrumConfiguration.__compute_complex_elements(this_complex, complex_features, idx)
                            for idx in range(len(complex_features))]
        return sum(complex_elements, [])

    """  ------------------------  Private supporting methods"""

    @staticmethod
    def __compute_complex_elements(this_complex: Complex,
                                   complex_features: List,
                                   index: int) -> List[FeaturedSimplex]:
        # Create simplicial element containing node indices associated with the simplex and feature set
        simplicial_node_feat = zip(this_complex.skeleton(index), np.array(complex_features[index]), strict=True)
        try:
            return [FeaturedSimplex(tuple(u), v) for u, v in simplicial_node_feat]
        except TypeError as e:
            print(e)
