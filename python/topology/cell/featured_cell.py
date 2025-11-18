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
from dataclasses import dataclass
from typing import List, Optional, AnyStr, Self
# 3rd Party imports
from toponetx.classes.cell import Cell
import numpy as np
# Library imports
__all__ = ['FeaturedCell']


@dataclass
class FeaturedCell:
    """
    Define a featured cell as a combination of a Cell (toponetx) and a features vector defined
    as a Numpy array
    @param cell: Toponetx
    @type cell: Cell
    @param features: Feature vector (dim=1)
    @type features: numpy.array
    """
    cell: Cell
    features: Optional[np.array] = None

    @classmethod
    def build(cls, indices: List[int], rank: int, features: np.array = None) -> Self:
        """
        Alternative constructor for a featured cell
        @param indices: List of node indices for this cell (edge, faces,...)
        @type indices: List[int]
        @param rank: Rank for this cell (0 for node, 1 for edge, 2 for face...)
        @type rank: int
        @param features: Feature vector (dim=1)
        @type features: numpy.array
        @return: Instance of a featured cell
        @rtype: FeaturedCell
        """
        if len(indices) < 2 or len(indices) > 32:
            raise ValueError(f"The number of indices {len(indices)} should [2, 32]")
        if rank < 0 or rank > 2:
            raise ValueError(f'Rank {rank} for featured cell is supported for [0, 2]')
        if 0 <= rank < 2 and len(indices) != rank+1:
            raise ValueError(f'Rank {rank} should be {len(indices)} -1')

        cell = Cell(elements=indices, rank=rank)
        return cls(cell=cell, features=features)

    def __str__(self) -> AnyStr:
        features_str = list(self.features) if self.features is not None else ''
        return f'\nElements: {self.cell.elements}\nFeatures: {features_str}'