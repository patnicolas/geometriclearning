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

from typing import FrozenSet, Optional, AnyStr, Tuple, Self
from dataclasses import dataclass

import numpy as np

from toponetx.classes.hyperedge import HyperEdge

@dataclass
class FeaturedHyperEdge:
    """
    Define a featured hyperedge or cell of a hypergraph as a combination of a TopoNetX hyperedge and a features vector
    defined as a Numpy array

    @param hyperedge: Toponetx hyperedge
    @type hyperedge: HyperEdge
    @param features: Feature vector (dim=1)
    @type features: numpy.array
    """
    hyperedge: HyperEdge
    features: np.array = None

    @classmethod
    def build(cls, hyperedge_indices: Tuple[int, ...], rank: int = None, features:  np.array = None) -> Self:
        return cls(HyperEdge(hyperedge_indices, rank), features)

    def get_indices(self) -> Tuple[int, ...]:
        indices: FrozenSet = self.hyperedge.elements
        return tuple(indices)

    def get_rank(self) -> int:
        return self.hyperedge.rank

    def __str__(self) -> AnyStr:
        features_str = str(self.features) if self.features is not None else 'NA'
        return f'Indices: {tuple(self.hyperedge.elements)} Features: {features_str}'
