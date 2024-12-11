__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from geomstats.geometry.pre_shape import PreShapeSpace
from geometry.visualization.space_visualization import VisualizationParams, SpaceVisualization
from geometry.geometric_space import GeometricSpace
import numpy as np
from typing import NoReturn, AnyStr


class KendallSpace(GeometricSpace):
    def __init__(self):
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
        :param num_samples Number of sample data points on the Kendall space
        :return Numpy array of random data points
        """
        return self.space.random_uniform(num_samples)

    @staticmethod
    def show(
            vParams: VisualizationParams,
            data_points: np.array,
            kendall_group_type: AnyStr) -> NoReturn:
        """
        Visualize the data points in 3D
        :param kendall_group_type: Type of Kendall group 'S32', 'M32'
        :type kendall_group_type: Str
        :param vParams Parameters for the visualization
        :param data_points Data points to visualize
        """
        space_visualization = SpaceVisualization(vParams)
        space_visualization.plot_3d(data_points, kendall_group_type)
