import math
import unittest

from Lie.Lie_SO3_group import LieSO3Group
from Lie.Lie_SO3_group import SO3Point
import numpy as np
from typing import AnyStr, List


class LieSO3GroupTest(unittest.TestCase):
    so3_rot_tgt_vectx = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
    so3_rot_tgt_vecty = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
    so3_rot_tgt_vectz = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


    @unittest.skip('Ignored')
    def test_init(self):
        so3_tangent_vec = LieSO3GroupTest.__create_tangent_vec('x', math.pi/2, LieSO3Group.identity)
        so3_tangent_vec = [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, -1.0]

        det = np.linalg.trace(np.reshape(so3_tangent_vec, (3, 3)))
        so3_group = LieSO3Group.build(so3_tangent_vec)
        print(str(so3_group))
        self.assertTrue(so3_group.group_element.shape == (3, 3))


    @unittest.skip('Ignored')
    def test_init_2(self):
        import numpy as np

        so3_tangent_vec = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        so3_group = LieSO3Group.build(so3_tangent_vec)
        so3_point1 = SO3Point(
            group_element=so3_group.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 point from tangent vector\n[1 0  0]\n[0 0 -1]\n[0 1  0]\nBase point: Identity')

        self.assertTrue(so3_group.group_element.shape == (3, 3))

        base_point = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        so3_group2 = LieSO3Group.build(so3_tangent_vec, base_point)
        so3_point2 = SO3Point(
            group_element=so3_group2.group_element,
            base_point=np.reshape(base_point, (3, 3)),
            descriptor='      Same SO3 point\nBase point:\n[0 -1  0]\n[0  0  0]\n[0  0  1]')
        print(str(so3_group2))
        LieSO3Group.visualize_all([so3_point1, so3_point2], 3)


    @unittest.skip('Ignored')
    def test_inverse(self):
        # Original SO3 rotation matrix
        so3_tangent_vec = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        so3_group = LieSO3Group.build(so3_tangent_vec)
        so3_point = SO3Point(
            group_element=so3_group.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 point from tangent vector\n[1 0  0]\n[0 0 -1]\n[0 1  0]\nBase point: Identity')

        print(str(so3_group))

        # Inverse SO3 rotation matrix
        so3_inv_group = so3_group.inverse()
        inv_so3_point = SO3Point(
            group_element=so3_inv_group.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 inverse point')
        print(f'SO3 Inverse point:{so3_inv_group}')
        self.assertTrue(so3_inv_group.group_element.shape == (3, 3))
        # Visualization
        LieSO3Group.visualize_all([so3_point, inv_so3_point], 2)


    @unittest.skip('Ignored')
    def test_product_1(self):
        # First SO3 rotation matrix 90 degree along x axis
        so3_tangent_vec = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec)
        so3_point = SO3Point(
            group_element=so3_group.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 point from tangent vector\n[1 0  0]\n[0 0 -1]\n[0 1  0]\nBase point: Identity')

        # Second SO3 rotation matrix
        so3_tangent_vec2 = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        so3_group2 = LieSO3Group.build(tgt_vector=so3_tangent_vec2)
        so3_point2 = SO3Point(
            group_element=so3_group2.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 point from tangent vector\n[0 -1 0]\n[1  0 0]\n[0  0 1]\nBase point: Identity')

        # Composition of two SO3 rotation matrices
        so3_group_product = so3_group.product(so3_group2)
        self.assertTrue(so3_group_product.group_element.shape == (3, 3))
        print(f'\nSO3 Product:{so3_group_product}')

        # Visualization
        LieSO3Group.visualize_all([so3_point, so3_point2], 0)
        so3_group_product.visualize('Composition of two SO3 matrices\nRotation along X with Z axis', 0)

    @unittest.skip('Ignored')
    def test_product_2(self):
        # First SO3 rotation matrix
        so3_tangent_vec = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec)
        so3_point = SO3Point(
            group_element=so3_group.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 point from tangent vector\n[1 0  0]\n[0 0 -1]\n[0 1  0]\nBase point: Identity')

        # Second SO3 rotation matrix
        identity_vec2 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        so3_group2 = LieSO3Group.build(tgt_vector=identity_vec2)
        so3_point2 = SO3Point(
            group_element=so3_group2.group_element,
            base_point=LieSO3Group.identity,
            descriptor='SO3 point from identity matrix\nBase point: Identity')

        # Composition of two SO3 rotation matrices
        so3_group_product = so3_group.product(so3_group2)
        self.assertTrue(so3_group_product.group_element.shape == (3, 3))
        print(f'\nSO3 points composition:{so3_group_product}')

        # Visualization
        LieSO3Group.visualize_all([so3_point, so3_point2], 0)
        so3_group_product.visualize('Composition of two SO3 matrices\n90 degree rotation with identity', 0)

    @unittest.skip('Ignored')
    def test_product_3(self):
        # First SO3 rotation matrix
        so3_tangent_vec = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec)
        so3_point = SO3Point(
            group_element=so3_group.group_element,
            base_point=LieSO3Group.identity,
            descriptor='3D Rotation:\n[0.4 0.3 0.8]\n[0.2 0.4 0.1]\n[0.1 0.2 0.6]')

        # Second SO3 rotation matrix
        so3_tangent_vec2 = [0.5]*len(so3_tangent_vec)
        so3_group2 = LieSO3Group.build(tgt_vector=so3_tangent_vec2)
        so3_point2 = SO3Point(
            group_element=so3_group2.group_element,
            base_point=LieSO3Group.identity,
            descriptor='3D Rotation:\n[0.5 0.5 0.5]\n[0.5 0.5 0.5]\n[0.5 0.5 0.5]')

        # Composition of two SO3 rotation matrices
        so3_group_product = so3_group.product(so3_group2)
        self.assertTrue(so3_group_product.group_element.shape == (3, 3))
        print(f'\nSO3 Product:{so3_group_product}')

        LieSO3Group.visualize_all([so3_point, so3_point2], 0)
        so3_group_product.visualize('Composition of 3D rotations')


    @unittest.skip('Ignored')
    def test_algebra(self):
        # First SO3 rotation matrix 90 degree along x axis
        so3_tangent_vec = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]

        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec)
        print(f'SO3 Rotation matrix: {so3_group}')
        lie_algebra = so3_group.lie_algebra()
        assert lie_algebra.size == len(so3_tangent_vec)
        print(f'\nLie algebra:\n{lie_algebra}')


    @unittest.skip('Ignored')
    def test_algebra2(self):
        # First SO3 rotation matrix 90 degree along x axis
        so3_tangent_vec = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]

        # Base point is SO3 rotation matrix 90 degree along y axis
        base_point = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec, base_point=base_point)
        print(f'SO3 point:\n{so3_group}')
        lie_algebra = so3_group.lie_algebra()
        print(f'\nLie algebra:\n{lie_algebra}')


    @unittest.skip('Ignored')
    def test_projection(self):
        # First SO3 rotation matrix 90 degree along x axis
        so3_tangent_vec = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]

        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec)
        projected = so3_group.projection()
        print(f'\nProjected point with identity:\n{projected.group_element}')

        # Base point is SO3 rotation matrix 90 degree along y axis
        base_point = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec, base_point=base_point)
        projected = so3_group.projection()
        print(f'\nProjected point with base point\n{so3_group.tangent_vec}:\n{projected.group_element}')


    def test_bracket(self):
        # First SO3 rotation matrix 90 degree along x axis
        so3_tangent_vec = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        so3_group = LieSO3Group.build(tgt_vector=so3_tangent_vec)
        print(f'\nTangent vector:\n{so3_group.tangent_vec}')
        np.set_printoptions(precision=3)
        print(f'\nSO3 point\n{so3_group.group_element}')
        bracket = so3_group.bracket(so3_tangent_vec)
        print(f'\nBracket [x,x]:\n{bracket}')


    @unittest.skip('Ignored')
    def test_bracket2(self):
        # First SO3 rotation matrix 90 degree along x axis
        so3_tangent_vec = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        so3_groupA = LieSO3Group.build(tgt_vector=so3_tangent_vec)
        print(f'\n{so3_groupA.group_element}')

        # Second SO3 rotation matrix 90 degree along y axis
        other_tgt_vec = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
        bracket = so3_groupA.bracket(other_tgt_vec)
        print(f'\nBracket:\n{bracket}')
        so3_groupB = LieSO3Group.build(tgt_vector=other_tgt_vec)

        so3_pointA = SO3Point(
            group_element = so3_groupA.tangent_vec,
            base_point = LieSO3Group.identity,
            descriptor='SO3 point from tangent vector\n[1 0  0]\n[0 0 -1]\n[0 1  0]\nBase point: Identity')
        print(f'SO3 point A:\n{so3_pointA.group_element}')
        so3_pointB = SO3Point(
            group_element = so3_groupB.tangent_vec,
            base_point = LieSO3Group.identity,
            descriptor='SO3 point from tangent vector\n[ 0 0 1]\n[ 0 1 0]\n[-1 0  0]\nBase point: Identity')
        print(f'SO3 point B:\n{so3_pointB.group_element}')
        LieSO3Group.visualize_all([so3_pointA, so3_pointB], 0)

        so3_point = SO3Point(
            group_element=bracket,
            base_point=LieSO3Group.identity,
            descriptor='Lie Bracket')
        LieSO3Group.visualize_all([so3_point], 0)


    @unittest.skip('Ignored')
    def test_visualize_geomstats(self):
        import geomstats.backend as gs
        import matplotlib.pyplot as plt
        from geomstats.geometry.special_orthogonal import SpecialOrthogonal
        import geomstats.visualization as visualization
        n_steps = 10

        so3_group = SpecialOrthogonal(n=3, point_type="vector")

        initial_point = so3_group.identity
        initial_tangent_vec = gs.array([0.5, 0.5, 0.8])
        geodesic = so3_group.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )

        t = gs.linspace(0.0, 1.0, n_steps)

        points = geodesic(t)
        visualization.plot(points, space="SO3_GROUP")
        plt.show()

    """ --------------------------  Private helper static methods -------------- """

    @staticmethod
    def __create_rotation_matrix(rotation_axis: AnyStr, theta: float) -> np.array:
        import math
        match rotation_axis:
            case 'x':
                return np.array(
                    [[1.0, 0.0, 0.0],
                     [0.0, math.cos(theta), -math.sin(theta)],
                     [0.0, math.sin(theta), math.cos(theta)]]
                )
            case 'y':
                return np.array(
                    [[math.cos(theta), 0.0, math.sin(theta)],
                     [0.0, 1.0, 0.0],
                     [-math.sin(theta), 0.0, math.cos(theta)]]
                )
            case 'z':
                return np.array(
                    [[math.cos(theta), -math.sin(theta), 0.0],
                     [math.sin(theta), math.cos(theta), 0.0],
                     [0.0, 0.0, 1.0]]
                )
            case _:
                raise Exception(f'Rotation axis {rotation_axis} is undefined')

    @staticmethod
    def __create_tangent_vec(rotation_axis: AnyStr, theta: float, base_point) -> List[float]:
        so3_matrix = LieSO3GroupTest.__create_rotation_matrix(rotation_axis, theta)
        return list(LieSO3Group.lie_group.log(so3_matrix, base_point))

