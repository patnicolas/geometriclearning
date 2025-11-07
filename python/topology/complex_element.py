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
from typing import AnyStr, List, Optional, Tuple, Dict
from dataclasses import dataclass
# 3rd Party imports
import numpy as np
# Library imports
from topology import TopologyException
__all__ = ['ComplexElement']


@dataclass
class ComplexElement:
    """
      Definition of the basic element of a Simplicial Complex {Node, Edge, Face} composed of
      - Feature vector
      - Indices of nodes defining this element

      @param node_indices: List of indices of nodes composing this simplicial element
      @type node_indices: List[int]
      @param feature_set: Feature vector or set associated with this simplicial element
      @type feature_set: Numpy array
      """
    node_indices: Tuple[int, ...] | None = None
    feature_set: Optional[np.array] = None

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
        if self.node_indices is None and override_node_indices is not None:
            self.node_indices = override_node_indices
        if self.node_indices is None:
            raise TopologyException('No node indices has been defined for this simplicial element')

        return tuple(self.node_indices), self.feature_set

    def __str__(self) -> AnyStr:
        output = []
        if self.feature_set is not None:
            output.append(list(np.round(self.feature_set, 5)))
        if self.node_indices is not None:
            output.append(self.node_indices)
        return ", ".join(map(str, output)) if len(output) > 0 else ""

