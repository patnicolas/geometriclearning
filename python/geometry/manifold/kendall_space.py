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
from typing import AnyStr
# 3rd Party imports
import numpy as np
from geomstats.geometry.pre_shape import PreShapeSpace
# Library imports
from geometry.visualization.space_visualization import VisualizationParams, SpaceVisualization
from geometry.manifold.geometric_space import GeometricSpace
__all__ = ['KendallSpace']


class KendallSpace(GeometricSpace):
    def __init__(self) -> None:
        m_ambient = 2
        k_landmarks = 3
        super(KendallSpace, self).__init__(m_ambient)

        GeometricSpace.manifold_type = 'KendallSphere'
        self.space = PreShapeSpace(m_ambient=m_ambient, k_landmarks=k_landmarks)
        self.space.equip_with_group_action("rotations")
        self.space.equip_with_quotient_structure()

    def sample(self, num_samples: int) -> np.array:
        """
        Generate random data on this Kendall space
        @param num_samples Number of sample data points on the Kendall space
        @return Numpy array of random data points
        """
        return self.space.random_uniform(num_samples)

    @staticmethod
    def show(
            vParams: VisualizationParams,
            data_points: np.array,
            kendall_group_type: AnyStr) -> None:
        """
        Visualize the data points in 3D
        @param kendall_group_type: Type of Kendall group 'S32', 'M32'
        @type kendall_group_type: Str
        @param vParams Parameters for the visualization
        @param data_points Data points to visualize
        """
        space_visualization = SpaceVisualization(vParams)
        space_visualization.plot_3d(data_points, kendall_group_type)
