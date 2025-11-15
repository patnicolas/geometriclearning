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
from typing import TypeVar, Generic
from abc import ABC, abstractmethod
# 3rd Party imports
import numpy as np
# Library imports
from topology.simplicial.featured_simplicial_elements import FeaturedSimplicialElements

__all__ = ['FeaturedComplex']
T = TypeVar('T')


class FeaturedComplex(ABC, Generic[T]):

    @abstractmethod
    def adjacency_matrix(self, directed_graph: bool = False) -> np.array:
        pass

    @abstractmethod
    def incidence_matrix(self, rank: int = 1, directed_graph: bool = True) -> np.array:
        pass

    @abstractmethod
    def laplacian(self, complex_laplacian: T) -> np.array:
        pass


