__author__ = "Patrick Nicolas"
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

# 3rd Party imports
import numpy as np
__all__ = ['EinSum']

class EinSum(object):
    def __init__(self, first: np.array, second: np.array = None) -> None:
        self.first = first
        self.second = second

    def dot(self) -> np.array:
        assert self.second is not None and \
                len(self.first.shape) == len(self.second.shape) == 1 and \
                self.first.shape[0] == self.first.shape[0]
        return np.einsum('i,i->', self.first, self.second)

    def matrix_mul(self) -> np.array:
        assert self.second is not None and \
                len(self.first.shape) == 2 \
                and self.first.shape == self.second.shape
        return np.einsum('ij,jk->ik', self.first, self.second)

    def matrix_el_sum(self) -> np.array:
        assert self.second is not None and \
                len(self.first.shape) == 2 \
                and self.first.shape == self.second.shape
        return np.einsum('ij,ij->', self.first, self.second)

    def outer_product(self) -> np.array:
        assert self.second is not None
        return np.einsum('i,j->ij', self.first, self.second)

    def transpose(self) -> np.array:
        assert len(self.first.shape) == 2
        return np.einsum('ij->ji', self.first)

    def trace(self) -> np.array:
        assert len(self.first.shape) == 2
        return np.einsum('ii->', self.first)

