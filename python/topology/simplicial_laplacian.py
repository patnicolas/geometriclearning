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

import toponetx as tnx
from typing import AnyStr, List
from enum import Enum
from dataclasses import dataclass
from topology import TopologyException
import numpy as np


class SimplicialLaplacianType(Enum):
    UpLaplacian = 'Upper-Laplacian'
    DownLaplacian = 'Lower-Laplacian'
    HodgeLaplacian = 'Hodge-Laplacian'


@dataclass
class SimplicialLaplacian:
    """
    Define the components of the Laplacian for Simplicial Complexes
    @param simplicial_laplacian_type Type of Laplacian (UP, DOWN or Hodge)
    @param rank Rank of the Laplacian
    @param signed Boolean flag to specify if the values of Laplacian are signed (Directed/Undirected)
    """
    simplicial_laplacian_type:  SimplicialLaplacianType
    rank: int
    signed: bool

    def __str__(self) -> AnyStr:
        return f'{self.simplicial_laplacian_type.value}, rank={self.rank}, signed={self.signed}'

    def __call__(self, simplicial_indices: List[List[int]]) -> np.array:
        """
        Compute the various combination of Laplacian (UP, DOWN, Hodge) for different rank.

        @param simplicial_indices: List of edge and face indices
        @type simplicial_indices: List of list
        @return: 2D Numpy array representing the Laplacian matrix
        @rtype: Numpy array
        """
        if len(simplicial_indices) < 1:
            raise TopologyException('Cannot compute simplicial Laplacian with undefined indices')

        try:
            sc = tnx.SimplicialComplex(simplicial_indices)
            match self.simplicial_laplacian_type:
                case SimplicialLaplacianType.UpLaplacian:
                    if self.rank < 0 or self.rank > 1:
                        raise TopologyException(f'Up-Laplacian does not support rank {self.rank }')
                    laplacian_matrix = sc.up_laplacian_matrix(self.rank, self.signed)

                case SimplicialLaplacianType.DownLaplacian:
                    if self.rank < 1 or self.rank > 2:
                        raise TopologyException(f'Down-Laplacian does not support rank {self.rank }')
                    laplacian_matrix = sc.down_laplacian_matrix(self.rank, self.signed)

                case SimplicialLaplacianType.HodgeLaplacian:
                    if self.rank < 0 or self.rank > 2:
                        raise TopologyException(f'Hodge-Laplacian does not support rank {self.rank}')
                    laplacian_matrix = sc.hodge_laplacian_matrix(self.rank, self.signed)

            return laplacian_matrix.toarray()
        except ValueError as e:
            raise TopologyException(e)
