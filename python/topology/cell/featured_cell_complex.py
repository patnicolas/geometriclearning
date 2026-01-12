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
from typing import List, AnyStr, TypeVar, Tuple
# 3rd Party imports
import toponetx as tnx
import numpy as np
# Library imports
from topology.cell.featured_cell import FeaturedCell
from topology.featured_complex import FeaturedComplex
from topology.complex_laplacian import ComplexLaplacian
__all__ = ['FeaturedCellComplex']
T = TypeVar('T')


class FeaturedCellComplex(FeaturedComplex[T]):
    """
    Implementation of featured cell complex:

    A cell complex is a broad, intuitive concept describing a space built from basic building blocks called cells
    (points, line segments, polygons, disks, etc.) that are glued together along their boundaries.
    Each k-cell is homeomorphic to an open k-dimensional ball
    The attaching maps describe how each higher-dimensional cell is connected to the lower-dimensional skeleton.
    The main goal is to represent spaces piecewise, using discrete, combinatorial components (cells).

    A cell complex (like a simplicial or combinatorial complex) is built from cells of different dimensions:
        0-cells: vertices
        1-cells: edges
        2-cells: faces
        3-cells: volumes, etc.

    The parameter type 'T' is the type of the first element of the representation of the components of this topological
    domain - Simplicial -> List[int],  Cell -> Cell, Hypergraph -> List[Tuple[int]]

    Reference Substack article: https://patricknicolas.substack.com/p/graphs-reimagined-the-power-of-cell

    Note: The implementation of the adjacency matrix is specific to graph. The TopoNetX library has a generic
    adjacency matrix to support edges - edges and faces -faces.
    """
    def __init__(self, featured_cells: frozenset[FeaturedCell]) -> None:
        """
        Constructor for a Featured Cell complex defined as a list of Featured cell {cell + feature vector}
        @param featured_cells: List of features cells
        @type featured_cells: List[FeaturedCell]
        """
        if len(featured_cells) == 0:
            raise ValueError('Cannot create a featured cell complex without featured cells')

        super(FeaturedCellComplex, self).__init__()
        self.featured_cells = featured_cells

    def __str__(self) -> AnyStr:
        return '\n'.join([str(f_cell) for f_cell in self.featured_cells])

    def adjacency_matrix(self, rank: Tuple[int, int] | int, signed: bool = False) -> np.array:
        """
        Computation of the adjacency matrix nodes - nodes
        Note: TopoNetX library has a generic adjacency matrix to support cells - cells of rank > 0
        @param rank: Rank of the Simplicial complex
        @type rank: int
        @param signed: Flag that specify if the graph is directed or not (Default Undirected graph)
        @type signed: bool
        @return: Adjacency matrix as a dense matrix
        @rtype: Numpy array
        """
        # Instantiate the cell complex as defined by
        cc = tnx.CellComplex(cells=[featured_cell.cell for featured_cell in self.featured_cells], signed=signed)
        # Initialize adjacency matrix
        n = len(set(cc.skeleton(rank=rank)))
        A = np.zeros(shape=(n, n), dtype=int)
        # Fill out adjacency
        for u, v in set(cc.skeleton(rank=rank+1)):
            A[u - 1, v - 1] = 1
            if signed:
                A[v - 1, u - 1] = 1
        return A

    def co_adjacency_matrix(self, rank: Tuple[int, int] | int, signed: bool = False) -> np.array:
        """
        Build the co-adjacency matrix for cells from any rank using the TopoNetX library
        @param rank: Rank of the cell 0, 1 & 2 are supported
        @type rank: int
        @param signed: Is the graph directed?
        @type signed: bool
        @return: CO-Adjacency matrix as a 2-dimensional Numpy array
        @rtype: Numpy array
        """
        if rank < 0 or rank > 2:
            raise ValueError(f'Rank of Co-adjacency matrix {rank} should be [0, 2]')

        cc = tnx.CellComplex(cells=[featured_cell.cell for featured_cell in self.featured_cells], signed=signed)
        return cc.coadjacency_matrix(rank=rank, signed=signed)

    def incidence_matrix(self, rank: Tuple[int, int] | int, signed: bool = True) -> np.array:
        """
        Extract the incidence matrix for a given rank and directed/undirected graph
        @param rank: Rank of the Simplicial complex
        @type rank: int
        @param signed: Flag that specifies if the graph is directed
        @type signed: bool
        @return: Incidence matrix
        @rtype: Numpy array
        """
        if rank < 0 or rank > 2:
            raise ValueError(f'Rank of incidence matrix {rank} should be [0, 2]')

        cc = tnx.CellComplex([featured_cell.cell for featured_cell in self.featured_cells])
        _, _, incidence = cc.incidence_matrix(rank=rank, index=True, signed=signed)
        return incidence.todense()

    def laplacian(self, cell_laplacian: ComplexLaplacian) -> np.array:
        """
        Computation of Up, Down, and Hodge Laplacian for Cell complex or rank 0, 1, and 2. This method invokes
        ComplexLaplacian.__call__ method

        @param cell_laplacian: Instance of the Laplacian for Cell complex
        @type cell_laplacian: ComplexLaplacian
        @return: Laplacian for this simplicial as 2-dimensional Numpy array
        @rtype: Numpy array
        """
        cells = [featured_cell.cell for featured_cell in self.featured_cells]
        return cell_laplacian(cells)
