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

from geomstats.geometry.euclidean import Euclidean
from geometry.geometric_space import GeometricSpace
from typing import NoReturn
import numpy as np
from geometry.visualization.space_visualization import VisualizationParams, SpaceVisualization

"""
    Define the Euclidean space and its components

    Class attributes:
    manifold_type: Type of manifold
    supported_manifolds: List of supported manifolds

    Object attributes:
    dimension: Dimension of this Euclidean space

    Methods:
    sample (pure abstract): Generate random data on a manifold
    show (abstract method): Display the Euclidean space in 3 dimension
"""

class EuclideanSpace(GeometricSpace):

    def __init__(self, dimension: int):
        super(EuclideanSpace, self).__init__(dimension)
        self.space = Euclidean(dim=self.dimension, equip=False)
        GeometricSpace.manifold_type = 'Euclidean'

    def sample(self, num_samples: int) -> np.array:
        """
        Generate random data on the Euclidean space
        :param num_samples Number of sample data points on the Euclidean space
        :return Numpy array of random data points
        """
        return self.space.random_point(num_samples)

    @staticmethod
    def show(vParams: VisualizationParams, data_points: np.array) -> NoReturn:
        """
        Visualize the data points in 3D
        :param vParams Parameters for the visualization
        :param data_points Data points to visualize
        """
        space_visualization = SpaceVisualization(vParams)
        space_visualization.plot_3d(data_points)
