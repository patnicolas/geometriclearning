__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import numpy as np
from typing import AnyStr, List, Self, Any
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
import geomstats.backend as gs
from geomstats.geometry.special_euclidean import SpecialEuclidean
import logging
logger = logging.getLogger('Lie.Lie_SE3_group')
__all__ = ['SE3Element', 'LieSE3Group']

@dataclass
class SE3Element:
    """
    Wrapper for Point or Matrix on SE3 manifold that leverages the Geomstats library.
    @param group_element  Point (3 x 3 matrix) on SE3 group
    @param rotation_matrix  Base point (3 coordinate) on SE3 group with 3x3 identity matrix as default,
                            Rotation A matrix in the Ax + B affine transformation)
    @param translation_matrix Description of the point
                            Translation B matrix (Ax + B affine transformation)
    """
    group_element: np.array
    rotation_matrix: np.array
    translation_matrix: np.array
    descriptor: AnyStr = 'SE3 Element'


class LieSE3Group(object):
    dim = 3
    lx_rotation = np.array([[0.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0]])
    ly_rotation = np.array([[0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [-1.0, 0.0, 0.0]])
    lz_rotation = np.array([[0.0, -1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0]])
    lx_translation = np.array([[1.0, 0.0, 0.0]])
    ly_translation = np.array([[0.0, 1.0, 0.0]])
    lz_translation = np.array([[0.0, 0.0, 1.0]])
    extend_rotation = np.array([[0.0, 0.0, 0.0]])
    extend_translation = np.array([[1.0]])

    def __init__(self, rotation_matrix: np.array, translation_matrix: np.array, epsilon: float = 0.001) -> None:
        """
        Constructor for the wrapper for key operations on SE3 Special Euclidean Lie manifold. A point on
        SE3 manifold is computed by composing the rotation and translation matrices
        @param rotation_matrix: 3 x 3 rotation matrix
        @type rotation_matrix: Numpy array
        @param translation_matrix: 1 x 3 translation matrix
        @type translation_matrix: Numpy array
        @param epsilon: Precision used for calculations involving potential division by 0 in rotations.
        @type epsilon: float
        """
        assert rotation_matrix.shape == (3, 3), f'Rotation matrix has incorrect shape {rotation_matrix.shape}'
        assert translation_matrix.shape == (1, 3), f'Translation matrix has incorrect shape {translation_matrix.shape}'
        assert 0 <= epsilon <= 0.2, f'Epsilon {epsilon} is out of range [0, 0.2]'

        self.lie_group = SpecialEuclidean(n=LieSE3Group.dim, point_type='matrix', epsilon=epsilon, equip=False)
        rotation_matrix, translation_matrix = LieSE3Group.reshape(rotation_matrix, translation_matrix)
        algebra_element = np.concatenate([rotation_matrix, translation_matrix], axis=1)
        self.group_element = self.lie_group.exp(algebra_element)

    @classmethod
    def build(cls,
              flatten_rotation_matrix: List[float],
              flatten_translation_vector: List[float],
              epsilon: float = 0.001) -> Self:
        """
        Build an instance of LieSE3Group given a rotation matrix, a tangent vector and a base point if defined
        @param flatten_rotation_matrix: 3x3 Rotation matrix (see. LieSO3Group)
        @type flatten_rotation_matrix: List[float]
        @param flatten_translation_vector: 3 length tangent vector for translation
        @type flatten_translation_vector: List[float]
        @param epsilon: Precision used for calculations involving potential division by 0 in rotations.
        @type epsilon: float
        @return: Instance of LieSE3Group
        @rtype: LieSE3Group
        """
        assert len(flatten_rotation_matrix) == 9, \
            f'The rotation matrix has {len(flatten_translation_vector)} elements. It should be 3'
        assert len(flatten_translation_vector) == 3, \
            f'Length of translation vector {len(flatten_translation_vector)} should be 3'

        np_rotation_matrix = np.reshape(flatten_rotation_matrix, (3, 3))
        np_translation_matrix = np.reshape(flatten_translation_vector, (1, 3))
        rotation_matrix, translation_matrix = LieSE3Group.reshape(np_rotation_matrix, np_translation_matrix)
        return cls(rotation_matrix, translation_matrix, epsilon)

    def inverse(self) -> Self:
        """
        Compute the inverse of this LieGroup element using Geomstats 'inverse' method
        @return: Instance of LieSE3Group
        @rtype: LieSE3Group
        """
        # Invoke Geomstats method
        inverse_group_element = self.lie_group.inverse(self.group_element)
        # Extract the 3x3 rotation matrix from the inverse
        rotation = inverse_group_element[:3, :3]
        # Extract the 1x3 translation matrix from the inverse element
        translation = np.expand_dims(inverse_group_element[:3, -1], axis=0)
        return LieSE3Group(rotation, translation)

    def compose(self, lie_se3_group: Self) -> Self:
        """
        Define the product this LieGroup point or element with another Lie group point using Geomstats compose method
        @param lie_se3_group Another Lie group
        @type LieSE3Group
        @return: Instance of LieSE3Group
        @rtype: LieSE3Group
        """
        a = self.lie_group.lie_algebra
        # Invoke Geomstats method
        composed_group_point = self.lie_group.compose(self.group_element,lie_se3_group.group_element)
        # Extract the 3x3 rotation matrix from the composed elements
        rotation = composed_group_point[:3, :3]
        # Extract the 1x3 translation matrix from the composed elements
        translation = np.expand_dims(composed_group_point[:3, -1], axis=0)
        return LieSE3Group(rotation, translation)

    def lie_algebra(self) -> Any:
        return self.lie_group.lie_algebra

    def log(self, group_element: np.array) -> np.array:
        return self.lie_group.log(group_element)

    def random_point_lie_algebra(self, num_samples: int, bound: int = 1.0) -> np.array:
        return self.lie_group.random_point(num_samples, bound)

    def belongs(self, matrix: np.array, atol: float = 1e-4) -> bool:
        return self.lie_group.belongs(matrix, atol)

    def __str__(self) -> AnyStr:
        matrix_str = '\n'.join([' '.join(f'{x:.3f}' for x in row) for row in self.group_element])
        return f'\nSE3 group element:\n{matrix_str}'

    def __repr__(self) -> AnyStr:
        return f'\nSE3 group element:\n{str(self.group_element)}'

    def visualize_tangent_space(self, rot_matrix: np.array, trans_vec: np.array) -> None:
        import matplotlib.pyplot as plt

        se3_element = SE3Element(self.group_element, rot_matrix, trans_vec)
        fig = plt.figure(figsize=(12, 12))
        fig.set_facecolor('#F2F9FE')
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.set_facecolor('#F2F9FE')
        title = f'Rotation matrix:\n{np.round(se3_element.rotation_matrix, 2)}'
        LieSE3Group.__visualize_element(se3_element.rotation_matrix, title, ax1)

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_facecolor('#F2F9FE')
        title = f'Translation matrix\n{np.round(se3_element.translation_matrix, 2)}'
        LieSE3Group.__visualize_element(se3_element.translation_matrix, title, ax2)

        plt.legend()
        plt.tight_layout()
        plt.show()

    def visualize(self, other: np.array, label: AnyStr) -> None:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 16))
        ax1 = fig.add_subplot(121, projection="3d")
        title = f'SE3 group element:\n{np.round(self.group_element, 2)}'
        LieSE3Group.__visualize_element(self.group_element, title, ax1)

        ax2 = fig.add_subplot(122, projection="3d")
        title = f'{label}:\n{np.round(other, 3)}'
        LieSE3Group.__visualize_element(other, title, ax2)

        plt.legend()
        plt.tight_layout()
        plt.show()

    """ ---------------------   Private Helper Methods --------------------------  """

    @staticmethod
    def reshape(rotation_matrix: np.array, translation_matrix: np.array) -> (np.array, np.array):
        """
        Constructor for the wrapper for key operations on SE3 Special Euclidean Lie manifold
        @param rotation_matrix: 3x3 Rotation matrix (see. LieSO3Group)
        @type rotation_matrix: Numpy array
        @param translation_matrix: 1x3 matrix for translation
        @type translation_matrix: Numpy array
        """
        rotation_matrix = gs.array(rotation_matrix)
        translation_matrix = gs.array(translation_matrix)
        rotation_matrix = np.concatenate([rotation_matrix, LieSE3Group.extend_rotation], axis=0)
        translation_matrix = np.concatenate([translation_matrix.T, LieSE3Group.extend_translation])
        return rotation_matrix,  translation_matrix

    @staticmethod
    def __visualize_element(se3_element: np.array, descriptor: AnyStr, ax: Axes3D) -> None:
        import geomstats.visualization as visualization

        visualization.plot(se3_element, ax=ax, space="SO3_GROUP")
        ax.set_title(descriptor, fontsize=14)
        LieSE3Group.__set_axes(ax)

    @staticmethod
    def __convert_translation_to_matrix(trans_vector: List[float]) -> np.array:
        zero_matrix = np.zeros((3, 3))
        for idx, diagonal in enumerate(trans_vector):
            zero_matrix[idx, idx] = diagonal
        return zero_matrix

    @staticmethod
    def __set_axes(ax: Axes3D) -> None:
        label_size = 11
        ax.set_xlabel('X values', fontsize=label_size)
        ax.set_ylabel('Y values', fontsize=label_size)
        ax.set_zlabel('Z values', fontsize=label_size)

        tick_size = 9
        for tick in ax.get_xticklabels():
            tick.set_fontsize(tick_size)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(tick_size)
        for tick in ax.get_zticklabels():
            tick.set_fontsize(tick_size)
