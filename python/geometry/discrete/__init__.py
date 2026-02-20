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
from abc import abstractmethod
# 3rd Party imports
import torch
__all__ = ['Wasserstein1Approximation', 'WassersteinException']

"""
Python package dedicated to Discrete Differential Geometry as element of Geometric Deep Learning
"""

class Wasserstein1Approximation(object):
    """
    Base class for Approximation of the Wasserstein distance.
    @see python/geometry/discrete/sinkhornknopp.py
    """
    __slots__ = ['r', 'c']

    def __init__(self, r: torch.Tensor, c: torch.Tensor) -> None:
        """
        Constructor for any approximation of the Wasserstein distance using the Marginal Distribution for the
        edges of a graph.
        Note: Approximation of the Wasserstein distance can be also computed using directly the Joint distribution
            on nodes - SinkhornKnopp.build

        @param r: Marginal probability distribution for rows of the Joint Distribution node - node
        @type r: torch.Tensor
        @param c: Marginal probability distribution for cols of the Joint Distribution node - node
        @type c: torch.Tensor
        """
        self.r = r
        self.c = c

    @abstractmethod
    def __call__(self, n_iters: int, early_stop_threshold: float) -> (int, torch.Tensor):
        pass


class WassersteinException(Exception):
    def __init__(self, *args, **kwargs):
        super(WassersteinException, self).__init__(args, kwargs)
