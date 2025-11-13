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
from typing import List, AnyStr
# 3rd Party imports
import toponetx as tnx
import numpy as np
# Library imports
from topology.cell.featured_cell import FeaturedCell
from topology.featured_complex import FeaturedComplex
from topology.complex_laplacian import ComplexLaplacian
__all__ = ['FeaturedCellComplex']


class FeaturedCellComplex(FeaturedComplex):
    def __init__(self, featured_cells: List[FeaturedCell]) -> None:
        super(FeaturedCellComplex, self).__init__()
        self.featured_cells = featured_cells

    def __str__(self) -> AnyStr:
        return '\n'.join([str(f_cell) for f_cell in self.featured_cells])

    def adjacency_matrix(self, directed_graph: bool = False) -> np.array:
        # Initialize adjacency matrix
        cc = tnx.CellComplex([featured_cell.cell for featured_cell in self.featured_cells])
        n = len(set(cc.skeleton(rank=0)))
        A = np.zeros((n, n), dtype=int)
        for u, v in set(cc.skeleton(rank=1)):
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

        cc = tnx.CellComplex([featured_cell.cell for featured_cell in self.featured_cells])
        _, _, incidence = cc.incidence_matrix(rank=rank, index=True, signed=directed_graph)
        return incidence.todense()

    def _validate(self) -> None:
        print('hello')

    def laplacian(self, simplicial_laplacian: ComplexLaplacian) -> np.array:
        simplicial_indices = [featured_cell.cell.elements for featured_cell in self.featured_cells]
        return simplicial_laplacian(simplicial_indices)
