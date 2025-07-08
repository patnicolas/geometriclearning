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

import numpy as np
from typing import AnyStr
from dataclasses import dataclass


class LieException(Exception):
    def __init__(self, *args, **kwargs):
        super(LieException, self).__init__(args, kwargs)


@dataclass
class LieElement:
    """
    Wrapper for Point or Matrix on SO3 manifold that leverages the Geomstats library.
    @param group_element Point on the Manifold
    @param identity_element  Base or reference point (i.g., Identity matrix)
    @param descriptor Description of the point

    Reference: https://patricknicolas.substack.com/p/practical-introduction-to-lie-groups
    """
    group_element: np.array     # Point on the Manifold
    identity_element: np.array  # Reference point (Identity by default)
    descriptor: AnyStr = 'Lie Element'


class UnitElements3D:
    x_rot = np.array([[0.0, 0.0, 0.0],   # Unit rotation along X axis
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
    y_rot = np.array([[0.0, 0.0, 1.0],    # Unit rotation along Y axis
                      [0.0, 0.0, 0.0],
                      [-1.0, 0.0, 0.0]])
    z_rot = np.array([[0.0, -1.0, 0.0],   # Unit rotation along Z axis
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])
    x_trans = np.array([1.0, 0.0, 0.0])  # Unit translation along X axis
    y_trans = np.array([0.0, 1.0, 0.0])  # Unit translation along Y axis
    z_trans = np.array([0.0, 0.0, 1.0])  # Unit translation along Z axis
    extend_rotation = np.array([[0.0, 0.0, 0.0]])
    extend_translation = np.array([[1.0]])


class UnitElements2D:
    x_rot = np.array([[0.0, -1.0],   # Unit rotation along X axis
                      [1.0, 0.0]])
    y_rot = np.array([[0.0, 1.0],    # Unit rotation along Y axis
                      [-1.0, 0.0]])

    x_trans = np.array([1.0, 0.0])  # Unit translation along X axis
    y_trans = np.array([1.0, 0.0])  # Unit translation along Y axis
    extend_rotation = np.array([[0.0, 0.0]])
    extend_translation = np.array([[1.0]])


u3d = UnitElements3D()
u2d = UnitElements2D()

