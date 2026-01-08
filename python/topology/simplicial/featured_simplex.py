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
from typing import AnyStr, List, Optional, Tuple, Dict
from dataclasses import dataclass
# 3rd Party imports
import numpy as np
# Library imports
from topology import TopologyException
__all__ = ['FeaturedSimplex']


@dataclass
class FeaturedSimplex:
    """
    Definition of the basic element of a Simplicial Complex {Node, Edge, Face} composed of
      - Feature vector
      - Indices of nodes defining this element (e.g., edge (3, 4), face (5, 7, 8)0

    @param simplex_indices: List of indices of nodes composing this simplicial element
    @type simplex_indices: List[int]
    @param features: Feature vector or set associated with this simplicial element
    @type features: Numpy array
    """
    simplex_indices: Tuple[int, ...] | None = None
    features: Optional[np.array] = None

    def get_rank(self) -> int:
        """
        Extract the rank of this simplicial complex from the number of the node indices in the simplex
        @return: Rank of the simplex
        @rtype: int
        """
        return len(self.simplex_indices)-1

    def __call__(self, override_node_indices: Tuple[int, ...] | None = None) -> Tuple[Tuple, np.array] | None:
        """
        Generate a tuple (node indices, feature vector) for this specific element. The node indices list is
        overridden only if it has not been already defined.
        A topology exception is raised if the node indices to be returned is None

        @param override_node_indices: Optional node indices
        @type override_node_indices: List[int]
        @return: Tuple (node indices, feature vector)
        @rtype: Tuple[Tuple, np.array]
        """
        if self.simplex_indices is None and override_node_indices is not None:
            self.simplex_indices = override_node_indices
        if self.simplex_indices is None:
            raise TopologyException('No node indices has been defined for this simplicial element')

        return tuple(self.simplex_indices), self.features

    def __str__(self) -> AnyStr:
        output = []
        if self.features is not None:
            output.append(list(np.round(self.features, 5)))
        if self.simplex_indices is not None:
            output.append(self.simplex_indices)
        return ", ".join(map(str, output)) if len(output) > 0 else ""

