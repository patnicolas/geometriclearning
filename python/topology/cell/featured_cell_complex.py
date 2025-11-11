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
from typing import Tuple
from abc import abstractmethod
# 3rd Party imports
import toponetx as tnx
import numpy as np
# Library imports
from topology.featured_complex import FeaturedComplex
__all__ = ['FeaturedCellComplex']


class FeaturedCellComplex(FeaturedComplex):
    def __init__(self, cells: Tuple[tnx.Cell, ...]) -> None:
        super(FeaturedCellComplex, self).__init__()
        self.cells = cells

    def adjacency_matrix(self, directed_graph: bool = False) -> np.array:
        # Initialize adjacency matrix
        n = len(np.concatenate([node.features for node in self.simplex_elements.featured_nodes]))
        A = np.zeros((n, n), dtype=int)

        # Fill in edges
        for u, v in [edge.simplex_indices for edge in self.simplex_elements.featured_edges]:
            A[u - 1, v - 1] = 1
            if directed_graph:
                A[v - 1, u - 1] = 1
        return A

    def incidence_matrix(self, rank: int = 1, directed_graph: bool = True) -> np.array:
        """
        Extract the incidence matrix for a given rank and directed/undirected graph
        @param rank: Rank of the Simplicial complex
        @type rank: int
        @param directed_graph: Flag that specifies if the graph is directed
        @type directed_graph: bool
        @return: Incidence matrix
        @rtype: Numpy array
        """
        if rank < 0 or rank > 2:
            raise ValueError(f'Rank of incidence matrix {rank} should be [0, 2]')

        sc = tnx.CellComplex(self.cells)
        _, _, incidence = sc.incidence_matrix(rank=rank, index=True, signed=directed_graph)
        return incidence.todense()

    @abstractmethod
    def _validate(self) -> None:
        pass

    @abstractmethod
    def laplacian(self, complex_laplacian: T) -> np.array:
        pass