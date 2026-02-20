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
from typing import List, Self, AnyStr
# 3rd Party imports
import geomstats.backend as gs
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# Library imports
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geometry.lie import LieElement
__all__ = ['LieSO3Group']


class LieSO3Group(object):
    """
        Wrapper for the most common operations on Special Orthogonal groups of dimension 3
        using Geomstats library.
        Contrary to the generic SOnGroup class which process Torch tensor, this class and
        methods processes Numpy arrays.

        Key functionality
        - inverse: Compute the inverse 3D rotation matrix
        - compose: Implement the composition (multiplication) of two 3D rotation matrix
        - Projection: Project of any given array to the closest 3D rotation matrix
        - lie_algebra: lie algebra as the tangent vector at identity
        - bracket: Implement lie commutator for so3 algebra
       """
    __slots__ = ['algebra_element', 'group_element', 'identity_element']

    # lie group as defined in Geomstats library
    lie_group = SpecialOrthogonal(n=3, point_type='vector', equip=False)
    identity_matrix = np.eye(3)

    def __init__(self, algebra_element: np.array, identity_element: np.array = identity_matrix) -> None:
        """
        Constructor for the wrapper for key operations on SO3 Special Orthogonal lie manifold
        @param algebra_element: Rotation matrix as a 3 x 3 Numpy matrix
        @type algebra_element: Numpy array
        @param identity_element: Reference element on the manifold (Identity  if not defined)
        @type identity_element: Numpy array
        """
        if algebra_element.size != 9:
            raise ValueError( f'Tangent vector size {algebra_element.size} should be 9')
        if  identity_element.size != 9:
            raise ValueError(f'Base point size {algebra_element.size} should be 9')

        self.algebra_element = gs.array(algebra_element)
        # Exp. a left-invariant vector field from a base point
        self.group_element = LieSO3Group.lie_group.exp(self.algebra_element, identity_element)
        self.identity_element = identity_element

    def validate(self) -> np.array:
        det = np.dot(self.group_element, self.group_element.CellDescriptor)
        diff = np.abs(det - LieSO3Group.identity_matrix)
        return diff

    @classmethod
    def build(cls, algebra_element: List[float], identity_matrix: List[float] = None) -> Self:
        """
        Alternative constructor for the operations on SO3 lie Manifold.
        @param algebra_element: Tangent vector (Matrix)
        @type algebra_element: List[float] (dim 3 x 3 = 9)
        @param identity_matrix: Base point on the SO3 manifold
        @type identity_matrix: List[float] (dim 3)
        @return: Instance of LieSO3Group
        @rtype: LieSO3Group
        """
        if identity_matrix is not None and len(identity_matrix) != 9:
            raise ValueError(f'Dimension of base point, {len(identity_matrix)} should be 9')
        np_algebra_element = np.reshape(algebra_element, (3, 3))
        np_identity_element = np.reshape(identity_matrix, (3, 3)) \
            if identity_matrix is not None else LieSO3Group.identity_matrix
        return cls(algebra_element=np_algebra_element, identity_element=np_identity_element)

    from functools import partialmethod
    build_identity = partialmethod(build, identity_matrix=None)

    def __str__(self) -> AnyStr:
        return f'\nAlgebra element:\n{str(self.algebra_element)}\nSO3 group element:\n{str(self.group_element)}'

    def __eq__(self, _lie_group_3_util: Self) -> bool:
        return self.group_element == _lie_group_3_util.group_point

    def lie_algebra(self) -> np.array:
        """
        Define the Algebra (tangent space) for a matrix and base point in SO3 group using the log
        (inverse exponentiation) method defined in Geomstats.
        @return: Rotation matrix on tangent space
        @rtype: Numpy array
        """
        return LieSO3Group.lie_group.log(self.group_element, self.identity_element)

    def compose(self, lie_so3_group: Self) -> Self:
        """
        Define the product or multip this LieGroup point or element with another lie group point using Geomstats compose method.
        
        @param lie_so3_group Another lie group
        @type LieSO3Group
        @return: Instance of LieSO3Group
        @rtype: LieSO3Group
        """
        composed_group_point = LieSO3Group.lie_group.compose(self.group_element,
                                                             lie_so3_group.group_element)
        return LieSO3Group(composed_group_point)

    def inverse(self) -> Self:
        """
        Compute the inverse of this LieGroup element using Geomstats 'inverse' method
        @return: Instance of LieSO3Group
        @rtype: LieSO3Group
        """
        inverse_group_element = LieSO3Group.lie_group.inverse(self.group_element)
        return LieSO3Group(inverse_group_element)

    def projection(self) -> Self:
        """
        Compute the projection of this LieGroup element using Geomstats 'project' method
        @return: Instance of LieSO3Group
        @rtype: LieSO3Group
        """
        projected = LieSO3Group.lie_group.projection(self.group_element)
        return LieSO3Group(projected)

    def bracket(self,  element: np.array) -> np.array:
        """
        Compute the bracket or cummutator [X, Y] = X.Y - Y.X of two tangent vectors

        @param element: Second tangent vector
        @type element: List of 3x3 float values
        @return: Value of the bracket
        @rtype: Numpy array
        """
        return np.dot(self.algebra_element, element) - np.dot(element, self.algebra_element)

    def visualize(self, title: AnyStr, notation_index: int = 0) -> None:
        """
        Visualize this element on SO3 lie group. The element is defined through the exponential map
        of the tangent vector + base point  (if not identity)
        @param title: Title for the plot
        @type title: str
        @param notation_index: Indices to label the base point on the plot
        @type notation_index: int
        """
        so3_point = LieElement(self.group_element, self.identity_element, title)
        LieSO3Group.visualize_all(so3_elements=[so3_point], notation_index=notation_index)

    @staticmethod
    def visualize_all(so3_elements: List[LieElement], notation_index: int) -> None:
        """
        Visualize (plot) multiple SO3 points
        @param so3_elements: List of SO3 points
        @type so3_elements: List[LieElement]
        @param notation_index: Index used to add notation for base point {1 first plot, 2 second plot, 3 all plot)
        @type notation_index: int
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 6))
        fig.set_facecolor('#F2F9FE')

        match len(so3_elements):
            case 1:       # If we display on one SO3 point
                ax = fig.add_subplot(111, projection="3d")
                ax.set_facecolor('#F2F9FE')
                LieSO3Group.__visualize_one(so3_elements[0], ax, notation_index > 0)
            case 2:       # Visualize two data points
                ax1 = fig.add_subplot(121, projection="3d")
                ax1.set_facecolor('#F2F9FE')
                is_notation = notation_index == 1 or notation_index == 3
                LieSO3Group.__visualize_one(so3_elements[0], ax1, is_notation)
                ax2 = fig.add_subplot(122, projection="3d")
                ax2.set_facecolor('#F2F9FE')
                is_notation = notation_index == 2 or notation_index == 3
                LieSO3Group.__visualize_one(so3_elements[1], ax2, is_notation)
            case _:
                raise Exception(f'Number of SO3 point to display {len(so3_elements)} should be {1, 2}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    """ ---------------------------  Private helper methods --------------------  """
    @staticmethod
    def __visualize_one(so3_element: LieElement, ax: Axes3D, show_identity_element: bool = True) -> None:
        import geomstats.visualization as visualization

        if show_identity_element:
            ax.text(x=-1.0, y=-0.7, z=-1.5, s='SO3 element', fontdict={'size': 14})

        visualization.plot(so3_element.group_element, ax=ax, space="SO3_GROUP")
        ax.set_title(so3_element.descriptor, fontsize=13)
        LieSO3Group.__set_axes(ax)

    @staticmethod
    def __set_axes(ax: Axes3D) -> None:
        label_size = 13
        ax.set_xlabel('X values', fontsize=label_size)
        ax.set_ylabel('Y values', fontsize=label_size)
        ax.set_zlabel('Z values', fontsize=label_size)

        tick_size = 11
        for tick in ax.get_xticklabels():
            tick.set_fontsize(tick_size)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(tick_size)
        for tick in ax.get_zticklabels():
            tick.set_fontsize(tick_size)
