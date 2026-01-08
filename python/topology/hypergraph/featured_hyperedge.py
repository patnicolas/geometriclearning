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

# Standard Library imports
from typing import FrozenSet, AnyStr, Self, Tuple
from dataclasses import dataclass
# Library imports
import numpy as np
from toponetx.classes.hyperedge import HyperEdge

@dataclass
class FeaturedHyperEdge:
    """
    Define a featured hyperedge or cell of a hypergraph as a combination of a TopoNetX hyperedge and a features vector
    defined as a Numpy array.
    This implementation leverages the class toponetx.classes.HyperEdge

    Note: Hyperedges can be labeled with any identifier. I use indices (int) start with 1 to stay consistent with
    my implementation of simplicial complex and cell complexes in the GitHub repository.

    @param hyperedge: Toponetx hyperedge
    @type hyperedge: HyperEdge
    @param features: Feature vector (dim=1)
    @type features: numpy array
    """
    hyperedge: HyperEdge
    features: np.array = None

    @classmethod
    def build(cls, hyperedge_indices: frozenset[int], rank: int = None, features:  np.array = None) -> Self:
        """
        Alternative constructor that builds a featured hyper from a tuple of node indices associated with the 
        hyperedge, rank and an optional features vector
        @param hyperedge_indices: Tuple of node indices associated with this hyperedge
        @type hyperedge_indices: Tuple[int, ...]
        @param rank: Rank of the hyperedge {0, 1, 2, 3,.)
        @type rank: int
        @param features: Optional features vector associated with this hyperedge
        @type features: Numpy array
        @return: Instance of FeaturedHyperEdge
        @rtype: FeaturedHyperEdge
        """
        return cls(HyperEdge(hyperedge_indices, rank), features)

    def get_indices(self) -> frozenset[int, ...]:
        """
        Retrieve the indices of the indices or labels associated with this featured hyperedge
        @return: Tuple of node indices associated with this hyperedge
        @rtype: Tuple[int, ...]
        """
        indices: FrozenSet = self.hyperedge.elements
        return indices

    def get_rank(self) -> int:
        """
        Retrieve the rank of this hyperedge as property of toponetx.classes.HyperEdge class
        @return: rank of this hyperedge
        @rtype: int
        """
        return self.hyperedge.rank

    def __str__(self) -> AnyStr:
        features_str = str(self.features) if self.features is not None else 'NA'
        return f'Indices: {tuple(self.hyperedge.elements)} Features: {features_str}'
