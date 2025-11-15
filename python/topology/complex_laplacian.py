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
from typing import AnyStr, List, Generic, TypeVar
# 3rd Party imports
import toponetx as tnx
from toponetx.classes.cell import Cell
from toponetx.classes.complex import Complex
import numpy as np
# Library imports
from topology import LaplacianType

CellDescriptor = TypeVar('CellDescriptor')
SimplexType = List
CellType = Cell
__all__ = ['SimplexType', 'CellType', 'ComplexLaplacian']


class ComplexLaplacian(Generic[CellDescriptor]):
    def __init__(self, laplacian_type:  LaplacianType, rank: int, signed: bool) -> None:
        """
        Constructor that defines the components of the Laplacian for Simplicial Complexes
        @param laplacian_type Type of Laplacian (UP, DOWN or Hodge)
        @param rank Rank of the Laplacian
        @param signed Boolean flag to specify if the values of Laplacian are signed (Directed/Undirected)
        """
        ComplexLaplacian.__validate(laplacian_type, rank)
        self.laplacian_type = laplacian_type
        self.rank = rank
        self.signed = signed

    def __str__(self) -> AnyStr:
        return f'{self.laplacian_type.value}, rank={self.rank}, signed={self.signed}'

    def __call__(self, complex_elements: CellDescriptor) -> np.array:
        """
        Compute the various combination of Laplacian (UP, DOWN, Hodge) for different rank.

        @param complex_elements: List of edge and face indices
        @type complex_elements: List of list
        @return: 2D Numpy array representing the Laplacian matrix
        @rtype: Numpy array
        """
        if len(complex_elements) < 1:
            raise ValueError('Cannot compute simplicial Laplacian with undefined indices')

        # Retrieve the appropriate Toponetx instance of cell or simplicial complex
        cplx = ComplexLaplacian.__get_complex(complex_elements)

        match self.laplacian_type:
            case LaplacianType.UpLaplacian:
                laplacian_matrix = cplx.up_laplacian_matrix(self.rank, self.signed)
            case LaplacianType.DownLaplacian:
                laplacian_matrix = cplx.down_laplacian_matrix(self.rank, self.signed)
            case LaplacianType.HodgeLaplacian:
                laplacian_matrix = cplx.hodge_laplacian_matrix(self.rank, self.signed)
        return laplacian_matrix.toarray()

    """  -------------------------  Private Helper Methods -------------------- """
    @staticmethod
    def __validate(laplacian_type:  LaplacianType, rank: int) -> None:
        if rank < 0 or rank > 2:
            raise ValueError(f'Rank {rank} is out of range')
        if laplacian_type == LaplacianType.UpLaplacian and rank > 1:
            raise ValueError(f'Rank {rank} for UP Laplacian is out-of-bounds')
        if laplacian_type == LaplacianType.DownLaplacian and (rank < 1 or rank > 2):
            raise ValueError(f'Rank {rank} for DOWN Laplacian is out-of-bounds')

    @staticmethod
    def __get_complex(complex_elements: CellDescriptor) -> Complex:
        if isinstance(complex_elements[0], SimplexType):
            cplx = tnx.SimplicialComplex(complex_elements)
        elif isinstance(complex_elements[0], CellType):
            cplx = tnx.CellComplex(complex_elements)
        else:
            raise TypeError(f'Type of Complex elements {complex_elements} is not supported')
        return cplx



