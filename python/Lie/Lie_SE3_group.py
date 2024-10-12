__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import numpy as np
from typing import AnyStr, List, Self, NoReturn
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
import geomstats.backend as gs
from geomstats.geometry.special_euclidean import SpecialEuclidean


"""
    Wrapper for Point or Matrix on SO3 manifold that leverages the Geomstats library.
    @param group_point  Point (3 x 3 matrix) on SO3 group
    @param base_point  Base point (3 coordinate) on SO3 group with 3x3 identity = as default
    @param descriptor Description of the point
"""


@dataclass
class SE3Point:
    group_element: np.array
    rotation_matrix: np.array
    translation_matrix: np.array


class LieSE3Group(object):
    dim = 3
    # Lie group as defined in Geomstats library
    lie_group = SpecialEuclidean(n=dim, point_type='matrix', epsilon=0.15, equip=False)
    extend_rotation = np.array([[0.0, 0.0, 0.0]])
    extend_translation = np.array([[1.0]])

    def __init__(self, se3_element: np.array) -> None:
        """
        Constructor for the wrapper for key operations on SE3 Special Euclidean Lie manifold. A point on
        SE3 manifold is computed by composing the rotation and translation matrices
        @param se3_element: 4 x 4 element in tangent space
        @type se3_element: Numpy array
        """
        self.se3_element = se3_element
        self.group_element = LieSE3Group.lie_group.exp(self.se3_element)

    @classmethod
    def build_from_numpy(cls, rot_matrix: np.array, trans_matrix: np.array) -> Self:
        """
        Constructor for the wrapper for key operations on SE3 Special Euclidean Lie manifold
        @param rot_matrix: 3x3 Rotation matrix (see. LieSO3Group)
        @type rot_matrix: Numpy array
        @param trans_matrix: 3x3 matrix for translation
        @type trans_matrix: Numpy array
        """
        assert rot_matrix.size == 9, f'Rotation matrix size {rot_matrix.size} should be 9'
        assert trans_matrix.size == 3, f'Translation matrix size {trans_matrix.size} should be 3'

        rotation_matrix = gs.array(rot_matrix)
        translation_matrix = gs.array(trans_matrix)
        se3_element = LieSE3Group.__build_se3_matrix(rotation_matrix, translation_matrix)
        return cls(se3_element)



    @classmethod
    def build_from_vec(cls, rot_matrix: List[float], trans_vector: List[float]) -> Self:
        """
        Build an instance of LieSE3Group given a rotation matrix, a tangent vector and a base point if defined
        @param rot_matrix: 3x3 Rotation matrix (see. LieSO3Group)
        @type rot_matrix: List[float]
        @param trans_vector: 3 length tangent vector for translation
        @type trans_vector: List[float]
        @return: Instance of LieSE3Group
        @rtype: LieSE3Group
        """
        assert len(rot_matrix) == 9, f'The rotation matrix has {len(trans_vector)} elements. It should be 3'
        assert len(trans_vector) == 3, f'Length of translation vector {len(trans_vector)} should be 3'

        np_rotation_matrix = np.reshape(rot_matrix, (3, 3))
        np_translation_matrix = np.reshape(trans_vector, (1, 3))
        return LieSE3Group.build_from_numpy(np_rotation_matrix, np_translation_matrix)

    def inverse(self) -> Self:
        """
        Compute the inverse of this LieGroup element using Geomstats 'inverse' method
        @return: Instance of LieSE3Group
        @rtype: LieSE3Group
        """
        inverse_group_point = LieSE3Group.lie_group.inverse(self.group_element)
        return LieSE3Group(inverse_group_point)

    def product(self, lie_se3_group: Self) -> Self:
        """
        Define the product this LieGroup point or element with another Lie group point using Geomstats compose method
        @param lie_se3_group Another Lie group
        @type LieSE3Group
        @return: Instance of LieSE3Group
        @rtype: LieSE3Group
        """
        composed_group_point = LieSE3Group.lie_group.compose(self.group_element, lie_se3_group.group_element)
        return LieSE3Group(composed_group_point)

    def __str__(self) -> AnyStr:
        return f'\nSE3 element:\n{str(self.se3_element)}\nLie group point:\n{str(self.group_element)}'

    def visualize_tangent_space(self, rot_matrix: List[float], trans_vec: List[float]) -> NoReturn:
        import matplotlib.pyplot as plt

        se3_point = self.__get_se3_point(rot_matrix, trans_vec)
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(121, projection="3d")
        title = f'Rotation matrix:\n{np.round(se3_point.rotation_matrix, 2)}'
        LieSE3Group.__visualize_element(se3_point.rotation_matrix, title, ax1)

        ax2 = fig.add_subplot(122, projection="3d")
        title = f'Translation matrix\n{np.round(se3_point.translation_matrix, 2)}'
        LieSE3Group.__visualize_element(se3_point.translation_matrix, title, ax2)

        plt.legend()
        plt.tight_layout()
        plt.show()


    def visualize(self, other: np.array, label: AnyStr) -> NoReturn:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 12))
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
    def __build_se3_matrix(rot_matrix: np.array, trans_matrix: np.array) -> np.array:
        extended_rot = np.concatenate([rot_matrix, LieSE3Group.extend_rotation], axis=0)
        extended_trans = np.concatenate([trans_matrix.T, LieSE3Group.extend_translation])
        return np.concatenate([extended_rot, extended_trans], axis=1)

    @staticmethod
    def __visualize_element(se3_element: np.array, descriptor: AnyStr, ax: Axes3D) -> NoReturn:
        import geomstats.visualization as visualization

        visualization.plot(se3_element, ax=ax, space="SO3_GROUP")
        ax.set_title(descriptor, fontsize=14)
        LieSE3Group.__set_axes(ax)

    def __get_se3_point(self,
                        rotation_matrix: List[float],
                        translation_vector: List[float]) -> SE3Point:
        np_rot_matrix = np.reshape(rotation_matrix, (3, 3))
        np_trans_matrix = np.reshape(translation_vector, (1, 3))
        return SE3Point(self.group_element, np_rot_matrix, np_trans_matrix)

    @staticmethod
    def __convert_translation_to_matrix(trans_vector: List[float]) -> np.array:
        zero_matrix = np.zeros((3, 3))
        for idx, diagonal in enumerate(trans_vector):
            zero_matrix[idx, idx] = diagonal
        return zero_matrix

    @staticmethod
    def __set_axes(ax: Axes3D) -> NoReturn:
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



"""
Certainly! Here's a simple example of how to work with the SE(3) Lie group and its associated Lie algebra using the Geomstats library in Python. SE(3) represents the special Euclidean group in 3D, which combines both rotations (SO(3)) and translations.

First, make sure you have installed the Geomstats library:

```bash
pip install geomstats
```

Now, here's the code that demonstrates basic operations with SE(3):

```python
import numpy as np
import geomstats.backend as gs
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.se3 import SE3

# Define the SE(3) group
se3_group = SE3()

# Example of an element of SE(3): combination of a rotation and a translation
rotation_matrix = gs.array([[0, -1, 0],
                            [1, 0, 0],
                            [0, 0, 1]])  # 90 degrees around z-axis
translation_vector = gs.array([1.0, 2.0, 3.0])  # Translation by (1, 2, 3)

se3_element = se3_group.compose(rotation_matrix, translation_vector)

# Print the SE(3) element
print("SE(3) element (rotation + translation):", se3_element)

# Lie algebra element (vector representation)
lie_algebra_element = gs.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
print("Lie algebra element (se3):", lie_algebra_element)

# Map the Lie algebra element to the Lie group (exponential map)
se3_from_algebra = se3_group.exp(lie_algebra_element)
print("SE(3) element from Lie algebra:", se3_from_algebra)

# Map back to the Lie algebra (logarithmic map)
lie_algebra_from_se3 = se3_group.log(se3_from_algebra)
print("Lie algebra element from SE(3):", lie_algebra_from_se3)

# Compose two elements of SE(3)
se3_element_2 = se3_group.compose(gs.eye(3), gs.array([0.5, 0.5, 0.5]))
se3_composed = se3_group.compose(se3_element, se3_element_2)
print("Composed SE(3) element:", se3_composed)

# Inverse of an SE(3) element
se3_inverse = se3_group.inverse(se3_element)
print("Inverse of SE(3) element:", se3_inverse)
```

### Explanation:
1. **SE(3) Group Definition:**
   We define the SE(3) group using the `SpecialEuclidean` class for dimension 3.

2. **Creating SE(3) Elements:**
   We create an SE(3) element by combining a rotation matrix and a translation vector.

3. **Lie Algebra:**
   We represent a Lie algebra element in vector form and map it to the SE(3) group using the exponential map (`exp`). The logarithmic map (`log`) takes us back to the algebra.

4. **Composing and Inverting Elements:**
   We demonstrate how to compose two SE(3) elements and how to compute the inverse of an SE(3) element.

This code provides a basic framework for working with SE(3) using Geomstats.



"""