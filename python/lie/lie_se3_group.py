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
from typing import AnyStr, List, Self
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
import geomstats.backend as gs
from geomstats.geometry.special_euclidean import SpecialEuclidean
__all__ = ['SE3Element', 'LieSE3Group', 'SE3ElementDescriptor']


@dataclass
class SE3ElementDescriptor:
    """
    Visualization of Algebra matrix with description in a given location for display
    @param vec: Tangent vector associated to SE3 algebra
    @type vec: Numpy Array
    @param x: X location of description of the SE3 algebra on plot
    @type x: float
    @param y: Y location of description of the SE3 algebra on plot
    @type y: float
    @param z: Z location of description of the SE3 algebra on plot
    @type z: float
    @param color: Color of descriptor of text
    @type color: str
    """
    vec: np.array
    x: float
    y: float
    z: float
    s: AnyStr
    color: AnyStr

    def draw(self, ax: Axes3D) -> None:
        ax.text(x=self.x,
                y=self.y,
                z=self.z,
                s=self.s,
                fontdict={'fontsize': 12, 'color': self.color},
                bbox=dict(facecolor='white', edgecolor='black'))


@dataclass
class SE3Element:
    """
    Wrapper for Point or Matrix on SE3 manifold that leverages the Geomstats library.
    @param group_element  Point (4 x 4 matrix) on SE3 group
    @param algebra_element  4x4 matrix at identity matrix as default with
                            Rotation A matrix in the Ax + B affine transformation)
    """
    algebra_element: np.array
    group_element: np.array
    descriptor: AnyStr = 'SE3 Element'

    def get_rotation(self) -> np.array:
        return self.algebra_element[0:3, 0:3]

    def __repr__(self) -> AnyStr:
        return (f'\n- SE3 Algebra:\n{self.algebra_element.astype(float)}'
                f'\n- SE3 Group:\n{self.group_element.astype(float)}')


class LieSE3Group(object):

    """
    Class Dedicated to LIE Special Euclidean group of dimension 3. Contrary to the generic
    SOnGroup class which process Torch tensor, this class and method processes Numpy arrays.
    This class leverage the Geomstats library.

    Key functionality
        - inverse: Compute the inverse 3D rotation matrix
        - compose: Implement the composition (multiplication) of two 3D rotation matrix
        - Projection: Project of any given array to the closest 3D rotation matrix
        - lie_algebra: lie algebra as the tangent vector at identity
        - bracket: Implement lie commutator for so3 algebra
    """
    def __init__(self,
                 rot_matrix: np.array,
                 trans_matrix: np.array,
                 epsilon: float = 0.001,
                 point_type: AnyStr = 'matrix') -> None:
        """
        Constructor for the wrapper for key operations on SE3 Special Euclidean lie manifold. A point on
        SE3 manifold is computed by composing the rotation and translation matrices

        @param rot_matrix: 3 x 3 rotation matrix
        @type rot_matrix: Numpy array
        @param trans_matrix: 1 x 3 translation matrix
        @type trans_matrix: Numpy array
        @param epsilon: Precision used for calculations involving potential division by 0 in rotations (default 1e-3)
        @type epsilon: float
        @param point_type: Representation of the SE3 element (default matrix)
        @type point_type: AnyStr
        """
        assert rot_matrix.shape == (3, 3), \
            f'Rotation matrix has incorrect shape {rot_matrix.shape}'
        assert trans_matrix.shape == (3,), \
            f'Translation matrix has incorrect shape {trans_matrix.shape}'
        assert 0 <= epsilon <= 0.2, f'Epsilon {epsilon} is out of range [0, 0.2]'
        assert point_type in ['matrix', 'vector'], f'Point type {point_type} should be matrix or vector'

        self.point_type = point_type
        self.lie_group = SpecialEuclidean(n=3, point_type=point_type, epsilon=epsilon, equip=True)

        algebra_element = np.eye(4)
        algebra_element[:3, :3] = rot_matrix
        algebra_element[:3, 3] = trans_matrix.flatten()

        # rotation_matrix, translation_matrix = LieSE3Group.reshape(rot_matrix, trans_matrix)
        # algebra_element = np.concatenate([rotation_matrix, translation_matrix], axis=1)
        self.se3_element = SE3Element(algebra_element, self.lie_group.exp(algebra_element))
        # Convert the (3, 3) Rotation matrix and (1, 3) Translation matrix into a (6, ) vector
        self.tangent_vector = LieSE3Group.get_tangent_vector(rot_matrix, trans_matrix)

    @classmethod
    def build(cls,
              flatten_rotation_matrix: List[float],
              flatten_translation_vector: List[float],
              epsilon: float,
              point_type: AnyStr) -> Self:
        """
        Build an instance of LieSE3Group given a rotation matrix, a tangent vector and a base point if defined
        @param flatten_rotation_matrix: 3x3 Rotation matrix (see. LieSO3Group)
        @type flatten_rotation_matrix: List[float]
        @param flatten_translation_vector: 3 length tangent vector for translation
        @type flatten_translation_vector: List[float]
        @param epsilon: Precision used for calculations involving potential division by 0 in rotations.
        @type epsilon: float
        @param point_type: Representation of the SE3 element (default matrix)
        @type point_type: AnyStr
        @return: Instance of LieSE3Group
        @rtype: SE3Visualization
        """
        assert len(flatten_rotation_matrix) == 3 * 3, \
            f'The rotation matrix has {len(flatten_translation_vector)} elements. It should be 3'
        assert len(flatten_translation_vector) == 3, \
            f'Length of translation vector {len(flatten_translation_vector)} should be 3'

        np_rotation_matrix = np.reshape(flatten_rotation_matrix, (3, 3))
        np_translation_matrix = np.reshape(flatten_translation_vector, (1, 3))
        rotation_matrix, translation_matrix = LieSE3Group.reshape(np_rotation_matrix, np_translation_matrix)
        return cls(rotation_matrix, translation_matrix, epsilon, point_type)

    from functools import partialmethod
    # Minimalist build method leveraging default values
    build_default = partialmethod(build, epsilon=0.001, point_type='matrix')

    @staticmethod
    def get_tangent_vector(rotation: np.array, translation: np.array) -> np.array:
        """
        Generates a (6, ) vector from a (3, 3) rotation matrix and (1, 3) translation matrix on tangent space.

        @param rotation: (3, 3) matrix
        @type rotation: Numpy array
        @param translation: (1, 3) matrix
        @type translation: Numpy array
        @return:  (6, ) vector representing 3 rotation elements + 3 translation element
        @rtype: Numpy array
        """
        trace = np.trace(rotation)
        theta = np.arccos(0.5*(trace-1))
        if np.isclose(theta, 0):
            omega = np.zeros(3)
        else:
            # Skew-symmetric part of the rotation
            omega_hat = (rotation - rotation.T) / (2 * np.sin(theta))
            omega = np.array([
                omega_hat[2, 1],
                omega_hat[0, 2],
                omega_hat[1, 0]
            ]) * theta

        return np.concatenate([omega, translation])

    def identity(self) -> np.array:
        return self.lie_group.identity

    def inverse(self) -> Self:
        """
        Compute the inverse of this LieGroup element using Geomstats 'inverse' method
        @return: Instance of LieSE3Group
        @rtype: SE3Visualization
        """
        # Invoke Geomstats method
        inverse_group_element = self.lie_group.inverse(self.se3_element.group_element)
        # Extract the 3x3 rotation matrix from the inverse
        rotation = inverse_group_element[:3, :3]
        # Extract the 1x3 translation matrix from the inverse element
        translation = np.array(inverse_group_element[:3, -1])
        return LieSE3Group(rot_matrix=rotation, trans_matrix=translation, point_type=self.point_type)

    def compose(self, lie_se3_group: Self) -> Self:
        """
        Define the product or composition of this LieGroup point or element with another
        lie group point using Geomstats compose method.

        @param lie_se3_group Another lie group
        @type lie_se3_group LieSE3Group
        @return: Instance of LieSE3Group
        @rtype: LieSE3Group
        """

        # Invoke Geomstats method
        composed_group_point = self.lie_group.compose(self.se3_element.group_element,
                                                      lie_se3_group.se3_element.group_element)
        # Extract the 3x3 rotation matrix from the composed elements
        rotation = composed_group_point[:3, :3]
        # Extract the 1x3 translation matrix from the composed elements
        translation = np.array(composed_group_point[:3, -1])
        return LieSE3Group(rot_matrix=rotation, trans_matrix=translation, point_type=self.point_type)

    def lie_algebra(self) -> np.array:
        return self.lie_group.lie_algebra

    def this_group_element(self) -> np.array:
        return self.se3_element.group_element

    def this_algebra_element(self) -> np.array:
        return self.se3_element.algebra_element

    def log(self, group_element: np.array) -> np.array:
        return self.lie_group.log(group_element)

    def random_point_lie_algebra(self, num_samples: int, bound: int = 1.0) -> np.array:
        return self.lie_group.random_point(num_samples, bound)

    def belongs(self, atol: float = 1e-4) -> bool:
        return self.lie_group.belongs(self.se3_element.group_element, atol).all()

    def jacobian_translation(self, matrix: np.array) -> np.array:
        return self.lie_group.jacobian_translation(matrix)

    def projection(self, matrix: np.array) -> np.array:
        return self.lie_group.regularize(matrix)

    def __str__(self) -> AnyStr:
        group_str = '\n'.join([' '.join(f'{x:.3f}' for x in row) for row in self.se3_element.group_element])
        algebra_str = '\n'.join([' '.join(f'{x:.3f}' for x in row) for row in self.se3_element.algebra_element])
        return f'\nSE3 group element:\n{group_str}\nSE3 Algebra\n{algebra_str}'

    def __repr__(self) -> AnyStr:
        return f'\nSE3 element:\n{str(self.se3_element)}'

    def visualize_tangent_space(self, rot_matrix: np.array, trans_vec: np.array) -> None:
        import matplotlib.pyplot as plt

        se3_element = SE3Element(self.se3_element.group_element, rot_matrix, trans_vec)
        fig = plt.figure(figsize=(12, 12))
        fig.set_facecolor('#F2F9FE')
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.set_facecolor('#F2F9FE')
        title = f'Rotation matrix:\n{np.round(se3_element.get_rotation(), 2)}'
        LieSE3Group.__visualize_element(se3_element.get_rotation(), title, ax1)

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_facecolor('#F2F9FE')
        title = f'Translation matrix\n{np.round(se3_element.get_rotation(), 2)}'
        LieSE3Group.__visualize_element(se3_element.get_rotation(), title, ax2)

        plt.legend()
        plt.tight_layout()
        plt.show()

    def visualize_element(self, other: np.array, label: AnyStr) -> None:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 16))
        ax1 = fig.add_subplot(121, projection="3d")
        title = f'SE3 group element:\n{np.round(self.se3_element.group_element, 2)}'
        LieSE3Group.__visualize_element(self.se3_element.group_element, title, ax1)

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
        Constructor for the wrapper for key operations on SE3 Special Euclidean lie manifold
        @param rotation_matrix: 3x3 Rotation matrix (see. LieSO3Group)
        @type rotation_matrix: Numpy array
        @param translation_matrix: 1x3 matrix for translation
        @type translation_matrix: Numpy array
        """
        from lie import u3d
        rotation_matrix = gs.array(rotation_matrix)
        translation_matrix = gs.array(translation_matrix)
        rotation_matrix = np.concatenate([rotation_matrix, u3d.extend_rotation], axis=0)
        translation_matrix = np.concatenate([translation_matrix.T, u3d.extend_translation])
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
