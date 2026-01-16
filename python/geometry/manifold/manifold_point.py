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
from typing import List, AnyStr, Optional, Self
from dataclasses import dataclass
# 3rd Party imports
import numpy as np
import torch
from geomstats.geometry.base import LevelSet
__all__ = ['ManifoldPoint']


@dataclass(frozen=True)
class ManifoldPoint:
    """
        Data class for defining a Manifold point as a pair of data point on the manifold and
        a vector.
        @param id: Identifier for the point on a manifold
        @param location: Point on the manifold
        @param tgt_vector: Optional reference Tangent vector
        @param geodesic: Enable computation and display of geodesic if True, none otherwise
        @param intrinsic: Flag if the coordinate for this Manifold point is intrinsic
    """
    id: AnyStr
    location: np.array
    tgt_vector: Optional[List[float]] = None
    geodesic: Optional[bool] = False
    intrinsic: Optional[bool] = False

    @classmethod
    def build(cls, _id: AnyStr, location: torch.Tensor) -> Self:
        return cls(_id, location.numpy())

    def ndim(self) -> int:
        return len(self.location)

    def to_intrinsic(self, space: LevelSet) -> np.array:
        """
        Convert the location as numpy array  from extrinsic to intrinsic coordinates if extrinsic
        coordinates have been used.
        The same coordinates are returned if the manifold location is intrinsic
        :param space Generic manifold
        :return intrinsic coordinates as Numpy array
        """
        return space.extrinsic_to_intrinsic_coords(self.location) if not self.intrinsic else self.location

    def to_intrinsic_polar(self, space: LevelSet) -> np.array:
        """
        Convert current extrinsic or intrinsic cartesian coordinates into intrinsic polar coordinates
        :param space Generic manifold
        :return intrinsic polar coordinates as Numpy array
        """
        intrinsic_coordinates = self.to_intrinsic(space)
        return ManifoldPoint.__cartesian_to_polar(intrinsic_coordinates)

    def to_extrinsic(self, space: LevelSet) -> np.array:
        """
        Convert the location as numpy array from intrinsic to extrinsic coordinates if intrinsic
        coordinates have been used.
        The same coordinates are returned if the manifold location is extrinsic
        :param space Generic manifold
        :return extrinsic coordinates as Numpy array
        """
        return space.intrinsic_to_extrinsic_coords(self.location) if self.intrinsic else self.location

    def __str__(self):
        tgt_vector_str = str(self.tgt_vector) if self.tgt_vector is not None else 'None'
        geodesic_str = 'Geodesic' if self.geodesic else 'No Geodesic'
        intrinsic_str = 'Intrinsic' if self.intrinsic else 'Extrinsic'
        return f'Id={self.id}, Base point={self.location}, Tangent Vector={tgt_vector_str},' \
               f'{geodesic_str}, {intrinsic_str}'

    # -------------- Helper private methods ----------------------

    @staticmethod
    def __cartesian_to_polar(cartesian_coordinates: np.array) -> np.array:
        import math
        # Make sure that the cartesian coordinates are defined in 2-dimension space
        if len(cartesian_coordinates) != 2:
            raise GeometricException(f'Number of coordinates {len(cartesian_coordinates)} should be 2')

        x = cartesian_coordinates[0]
        y = cartesian_coordinates[1]

        # The intrinsic cartesian point cannot be the origin
        if x == 0.0 and y == 0.0:
            raise GeometricException(f'x {x} and y {y} should not be 0')

        r = math.sqrt(x ** 2 + y ** 2)
        theta = math.acos(x / r) if y >= 0.0 else -math.acos(x / r)
        return np.array([r, theta])
