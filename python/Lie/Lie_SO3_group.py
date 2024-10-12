__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from typing import List, Self, AnyStr, Tuple, NoReturn
from dataclasses import dataclass


"""
    Wrapper for Point or Matrix on SO3 manifold that leverages the Geomstats library.
    @param group_point  Point (3 x 3 matrix) on SO3 group
    @param base_point  Base point (3 coordinate) on SO3 group with 3x3 identity = as default
    @param descriptor Description of the point
"""


@dataclass
class SO3Point:
    group_element: np.array
    base_point: np.array
    descriptor: AnyStr


"""
    Wrapper for the most common operations on SO3 groups using Geomstats library
    - inverse: Compute the inverse 3D rotation matrix
    - product: Implement the composition (multiplication) of two 3D rotation matrix
    - Projection: 
"""


class LieSO3Group(object):
    dim = 3
    # Lie group as defined in Geomstats library
    lie_group = SpecialOrthogonal(n=dim, point_type='vector', equip=False)
    identity = gs.eye(dim)

    def __init__(self, tgt_vector: np.array, base_point: np.array = identity) -> None:
        """
        Constructor for the wrapper for key operations on SO3 Special Orthogonal Lie manifold
        @param tgt_vector: Tangent vector as a 3 x 3 Numpy matrix
        @type tgt_vector: Numpy array
        @param base_point: Base point vector on the manifold (Identity [0, 0, 0] if not defined)
        @type base_point: Numpy array
        """
        assert tgt_vector.size == 9, f'Tangent vector size {tgt_vector.size} should be 9'
        assert base_point.size == 9, f'Base point size {tgt_vector.size} should be 9'

        self.tangent_vec = gs.array(tgt_vector)
        # Exp. a left-invariant vector field from a base point
        self.group_element = LieSO3Group.lie_group.exp(self.tangent_vec, base_point)
        self.base_point = base_point

    def validate(self) -> np.array:
        det = np.dot(self.group_element, self.group_element.T)
        diff = np.abs(det - LieSO3Group.identity)
        return diff

    @classmethod
    def build(cls, tgt_vector: List[float], base_point: List[float] = None) -> Self:
        """
        Alternative constructor for the operations on SO3 Lie Manifold
        @param tgt_vector: Tangent vector (Matrix)
        @type tgt_vector: List[float] (dim 3 x 3 = 9)
        @param base_point: Base point on the SO3 manifold
        @type base_point: List[float] (dim 3)
        @return: Instance of LieSO3Group
        @rtype: LieSO3Group
        """
        assert base_point is None or len(base_point) == 9, \
            f'Dimension of base point, {len(base_point)} should be 9'

        np_tgt_vector = np.reshape(tgt_vector, (3, 3))
        np_point = np.reshape(base_point, (3, 3)) if base_point is not None else LieSO3Group.identity
        return cls(tgt_vector=np_tgt_vector, base_point=np_point)

    def __str__(self) -> AnyStr:
        return f'\nTangent vector:\n{str(self.tangent_vec)}\nLie group point:\n{str(self.group_element)}'

    def __eq__(self, _lie_group_3_util: Self) -> bool:
        return self.group_element == _lie_group_3_util.group_point

    def lie_algebra(self) -> np.array:
        """
        Define the Algebra (tangent space) for a matrix and base point in SO3 group using the log
        (inverse exponentiation) method defined in Geomstats
        @return: Rotation matrix on tangent space
        @rtype: Numpy array
        """
        return LieSO3Group.lie_group.log(self.group_element, self.base_point)

    def product(self, lie_so3_group: Self) -> Self:
        """
        Define the product this LieGroup point or element with another Lie group point using Geomstats compose method
        @param lie_so3_group Another Lie group
        @type LieSO3Group
        @return: Instance of LieSO3Group
        @rtype: LieSO3Group
        """
        composed_group_point = LieSO3Group.lie_group.compose(self.group_element, lie_so3_group.group_element)
        return LieSO3Group(composed_group_point)

    def inverse(self) -> Self:
        """
        Compute the inverse of this LieGroup element using Geomstats 'inverse' method
        @return: Instance of LieSO3Group
        @rtype: LieSO3Group
        """
        inverse_group_point = LieSO3Group.lie_group.inverse(self.group_element)
        return LieSO3Group(inverse_group_point)

    def projection(self) -> Self:
        """
        Compute the projection of this LieGroup element using Geomstats 'project' method
        @return: Instance of LieSO3Group
        @rtype: LieSO3Group
        """
        projected = LieSO3Group.lie_group.projection(self.group_element)
        return LieSO3Group(projected)

    def bracket(self,  _tgt_vector: List[float]) -> np.array:
        """
        Compute the bracket [X, Y] = X.Y - Y.X of two tangent vectors
        @param _tgt_vector: Second tangent vector
        @type _tgt_vector: List of 3x3 float values
        @return: Value of the bracket
        @rtype: Numpy array
        """
        assert len(_tgt_vector) == 9, f'Rotation matrix size {len(_tgt_vector)} should be 9'

        np_tgt_vector = np.reshape(_tgt_vector, (3, 3))
        return np.dot(self.tangent_vec, np_tgt_vector) - np.dot(np_tgt_vector, self.tangent_vec)

    def visualize(self, title: AnyStr, notation_index: int = 0) -> NoReturn:
        """
        Visualize this element on SO3 Lie group. The element is defined through the exponential map
        of the tangent vector + base point  (if not identity)
        @param title: Title for the plot
        @type title: str
        @param notation_index: Indices to label the base point on the plot
        @type notation_index: int
        """
        so3_point = SO3Point(self.group_element, self.base_point, title)
        LieSO3Group.visualize_all([so3_point], notation_index)

    @staticmethod
    def visualize_all(so3_points: List[SO3Point], notation_index: int) -> NoReturn:
        """
        Visualize (plot) multiple SO3 points
        @param so3_points: List of SO3 points
        @type so3_points: List[SO3Point]
        @param notation_index: Index used to add notation for base point {1 first plot, 2 second plot, 3 all plot)
        @type notation_index: int
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 12))

        match len(so3_points):
            case 1:       # If we display on one SO3 point
                ax = fig.add_subplot(111, projection="3d")
                LieSO3Group.__visualize_one(so3_points[0], ax, notation_index > 0)
            case 2:       # Visualize two data points
                ax1 = fig.add_subplot(121, projection="3d")
                is_notation = notation_index == 1 or notation_index == 3
                LieSO3Group.__visualize_one(so3_points[0], ax1, is_notation)
                ax2 = fig.add_subplot(122, projection="3d")
                is_notation = notation_index == 2 or notation_index == 3
                LieSO3Group.__visualize_one(so3_points[1], ax2, is_notation)
            case _:
                raise Exception(f'Number of SO3 point to display {len(so3_points)} should be {1, 2}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    """ ---------------------------  Private helper methods --------------------  """
    @staticmethod
    def __visualize_one(so3_point: SO3Point, ax: Axes3D, show_base_point: bool = True) -> NoReturn:
        import geomstats.visualization as visualization

        if show_base_point:
            ax.text(x=-1.3, y=-0.7, z=-0.5, s='Base point', fontdict={'size': 14})

        visualization.plot(so3_point.group_element, ax=ax, space="SO3_GROUP")
        ax.set_title(so3_point.descriptor, fontsize=15)
        LieSO3Group.__set_axes(ax)

    @staticmethod
    def __set_axes(ax: Axes3D) -> NoReturn:
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






    """
     Create a vector in the Lie algebra
tangent_vec = gs.array([0.1, 0.2, 0.3])

# Use the exponential map to get a point on the Lie group (SO(3))
point_on_group = SO3.exp(tangent_vec)
print("Point on SO(3):", point_on_group)

# Use the logarithmic map to get back to the Lie algebra
log_map = SO3.log(point_on_group)
print("Log map (Lie algebra):", log_map)


# Create another point on the Lie group
point_on_group2 = SO3.exp(gs.array([0.3, -0.2, 0.5]))

# Perform group multiplication (composition of rotations)
group_product = SO3.compose(point_on_group, point_on_group2)
print("Group multiplication (composition):", group_product)

# Compute the inverse of a point on the Lie group
group_inverse = SO3.inverse(point_on_group)
print("Inverse of the point on SO(3):", group_inverse)


# Create a vector in the Lie algebra of SE(3)
tangent_vec_se3 = gs.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 3 for rotation, 3 for translation

# Exponential map to get a point on SE(3) (translation and rotation)
point_on_se3 = SE3.exp(tangent_vec_se3)
print("Point on SE(3):", point_on_se3)

# Logarithmic map to return to Lie algebra
log_map_se3 = SE3.log(point_on_se3)
print("Log map (SE(3)): ", log_map_se3)
    
    # Random sampling from SO(3)
random_point_so3 = SO3.random_point()
print("Random point on SO(3):", random_point_so3)

# Compute the Riemannian distance between two points on the group
distance_so3 = SO3.metric.dist(point_on_group, point_on_group2)
print("Distance between two points on SO(3):", distance_so3)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

visualization.plot(data[:2], ax=ax, space="SO3_GROUP")

ax.set_title("3D orientations of the beds.");
    """