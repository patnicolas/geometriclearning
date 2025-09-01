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
from typing import List
import logging
# Library imports
from geometry.manifold.manifold_point import ManifoldPoint
from geometry.manifold.hypersphere_space import HypersphereSpace
__all__ = ['GeometricDistribution']



class GeometricDistribution(object):
    """
    Define a generic Geometric Distribution on an Hypersphere using the Geomstats Python library
    The purpose of this class is to display data points and associated tangent vectors on an
    Hypersphere as defined in the class HypersphereSpace.
    """
    _ZERO_TGT_VECTOR = (0.0, 0.0, 0.0)

    def __init__(self) -> None:
        """
        Constructor for the generic geometric distribution on a hypersphere
        """
        self.manifold = HypersphereSpace(True)

    def show_points(self,
                    num_pts: int,
                    tgt_vector: List[float] = _ZERO_TGT_VECTOR) -> int:
        """
        Display the data points on a manifold (Hypersphere). The tangent vector is displayed if
        is not defined as the extrinsic origin zero_tgt_vector = (0.0, 0.0, 0.0)

        @param num_pts: Number of points to be displayed on Hypersphere
        @type num_pts: int
        @param tgt_vector: Tangent vector extrinsic coordinate
        @type tgt_vector: List of float
        @return: Number of points from exponential map
        @rtype: int
        """
        manifold_pts = self._random_manifold_points(num_pts, tgt_vector)
        exp_map = self.manifold.tangent_vectors(manifold_pts)
        for tangent_vector, end_pt in exp_map:
            logging.info(f'{tangent_vector=},{end_pt=}')

        self.manifold.show_manifold(manifold_pts)
        return len(manifold_pts)

    """ --------------------  Protected Helper Method ---------------------------  """

    def _random_manifold_points(self, num_pts: int, tgt_vector: List[float]) -> List[ManifoldPoint]:
        p = self.manifold.sample(num_pts)
        return [
            ManifoldPoint(
                id=f'data{index}',
                location=sample,
                tgt_vector=tgt_vector,
                geodesic=False) for index, sample in enumerate(p)
        ]
