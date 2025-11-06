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
from typing import AnyStr, List, Tuple
from dataclasses import dataclass
# 3rd Party imports
from toponetx.classes.complex import Complex
import numpy as np
# Library imports
from topology.simplicial.abstract_simplicial_complex import ComplexElement
from topology.graph_complex_elements import GraphComplexElements


@dataclass
class HodgeLaplacianEigenvectors:
    """
    Class to generate a set of complex elements from a graph using the Hodge Laplacian

    @param num_eigenvectors: Number of eigenvectors or eigenvalues used for each of the Indidence matrices for
                    nodes, edges and simplex-2
    @type num_eigenvectors: int
    """
    num_eigenvectors: Tuple[int, int, int]

    def get_num_node_eigenvalues(self) -> int:
        return self.num_eigenvectors[0]

    def get_num_edge_eigenvalues(self) -> int:
        return self.num_eigenvectors[1]

    def get_num_simplex_2_eigenvalues(self) -> int:
        return self.num_eigenvectors[2]

    def __str__(self) -> AnyStr:
        return (f'\n{self.num_eigenvectors[0]} node eigenvalues {self.num_eigenvectors[1]} edges eigenvalues '
                f'{self.num_eigenvectors[2]} edges eigenvalues')

    def get_complex_features(self, this_complex: Complex) -> GraphComplexElements:
        """
        Extract the simplex features for nodes, edges and simplex_2

        Steps
        1: Add features values to each node
        2: Add features to edges
        3: Add features to faces

        @param this_complex: This simplicial or cell complex
        @type this_complex: Complex
        @return: Fully configured elements of the simplicial complex
        @rtype: AbstractSimplicialComplex
        """
        from toponetx.algorithms.spectrum import hodge_laplacian_eigenvectors

        # Compute the laplacian weights for nodes, edges (L1) and faces (L2)
        complex_features = \
            [hodge_laplacian_eigenvectors(this_complex.hodge_laplacian_matrix(idx), self.num_eigenvectors[idx])[1]
             for idx in range(len(self.num_eigenvectors))]

        # Generate the simplex related to node, edge and simplex_2 (triangles, cells ...)
        complex_elements = [HodgeLaplacianEigenvectors.__compute_complex_elements(this_complex, complex_features, idx)
                            for idx in range(len(complex_features))]
        return GraphComplexElements.build(complex_elements)

    @staticmethod
    def __compute_complex_elements(this_complex: Complex,
                                   complex_features: List,
                                   index: int) -> List[ComplexElement]:
        # Create simplicial element containing node indices associated with the simplex and feature set
        simplicial_node_feat = zip(this_complex.skeleton(index), np.array(complex_features[index]), strict=True)
        return [ComplexElement(tuple(u), v) for u, v in simplicial_node_feat]
