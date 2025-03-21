import math
import unittest

from Lie.Lie_SO3_group import LieSO3Group
from Lie.Lie_SO3_group import LieElement
import numpy as np
from typing import AnyStr, List


class LieSO3GroupTest(unittest.TestCase):
    so3_rot_tgt_vectx = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
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
        ly_element = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]
        so3_group = LieSO3Group.build(ly_element)
        so3_element_1 = LieElement(
            group_element=so3_group.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 Ly element\n[0 0  1]\n[0 0 0]\n[-1 0  0]\n@ identity element')

        self.assertTrue(so3_group.group_element.shape == (3, 3))

        lz_element = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        so3_group2 = LieSO3Group.build(lz_element)
        so3_element_2 = LieElement(
            group_element=so3_group2.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 Lz element\nIdentity element:\n[0 -1  0]\n[1  0  0]\n[0  0  0]')
        print(str(so3_group2))
        LieSO3Group.visualize_all(so3_elements=[so3_element_1, so3_element_2], notation_index=3)

    @unittest.skip('Ignored')
    def test_init_3(self):
        lx_element = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        ly_element = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]
        lz_element = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cx = 2.0
        cy = -0.5
        cz = 1.5

        element = cx*np.array(lx_element) + cy*np.array(ly_element) + cz*np.array(lz_element)
        element = element.reshape((3, -1))
        print(element)
        so3_group = LieSO3Group(element)
        so3_element = LieElement(
            group_element=so3_group.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 element for\n[0.0 -1.5 -0.5]\n[1.5  0.0 -2.0]\n[0.5  2.0  0.0]\n@ identity element')
        LieSO3Group.visualize_all(so3_elements=[so3_element], notation_index=3)

    @unittest.skip('Ignored')
    def test_inverse(self):
        # Generator for rotation along Y axis
        ly_element = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]
        so3_group = LieSO3Group.build(ly_element)
        element = LieElement(
            group_element=so3_group.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 point from tangent vector\n[1 0  0]\n[0 0 -1]\n[0 1  0]\nBase point: Identity')

        print(str(so3_group))

        # Inverse SO3 rotation matrix
        so3_inv_group = so3_group.inverse()
        inv_so3_inverse = LieElement(
            group_element=so3_inv_group.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 inverse point')
        print(f'SO3 Inverse point:{so3_inv_group}')
        self.assertTrue(so3_inv_group.group_element.shape == (3, 3))
        # Visualization
        LieSO3Group.visualize_all([element, inv_so3_inverse], 2)

    @unittest.skip('Ignored')
    def test_product_1(self):
        # First SO3 rotation matrix along X axis
        lx_element = np.array([[0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0],
                               [0.0, 1.0, 0.0]])
        so3_group_x = LieSO3Group(algebra_element=lx_element)
        so3_element_x = LieElement(
            group_element=so3_group_x.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 Rotation along X\n[0 0  0]\n[0 0 -1]\n[0 1  0]\n@ identity')

        # Second SO3 rotation along Y axis
        ly_element = np.array([[0.0, 0.0, 1.0],
                               [0.0, 0.0, 0.0],
                               [-1.0, 0.0, 0.0]])
        so3_group_y = LieSO3Group(algebra_element=ly_element)
        so3_point_y = LieElement(
            group_element=so3_group_y.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 Rotation along Y\n[ 0 0 1]\n[ 0 0 0]\n[-1 0 0]\n@ identity')

        # Composition of two SO3 rotation matrices
        so3_group_product = so3_group_x.product(so3_group_y)
        self.assertTrue(so3_group_product.group_element.shape == (3, 3))
        print(f'\nSO3 Product:{so3_group_product}')

        # Visualization
        LieSO3Group.visualize_all([so3_element_x, so3_point_y], 0)
        so3_group_product.visualize('Composition of two SO3 matrices\nRotation along X with Y axis', 0)

    @unittest.skip('Ignored')
    def test_product_2(self):
        # First SO3 rotation matrix along X axis
        lx_element = np.array([[0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0],
                               [0.0, 1.0, 0.0]])
        so3_group_x = LieSO3Group(algebra_element=lx_element)
        so3_element_x = LieElement(
            group_element=so3_group_x.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 Rotation along X\n[0 0  0]\n[0 0 -1]\n[0 1  0]\n@ identity')

        # Second SO3 rotation along Y axis
        identity = np.eye(3)

        so3_group_identity = LieSO3Group(algebra_element=identity)
        so3_point_identity = LieElement(
            group_element=so3_group_identity.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='Identity\n[1 0 0]\n[0 1 0]\n[0 0 1]\n@ identity')

        # Composition of two SO3 rotation matrices
        so3_group_product = so3_group_x.product(so3_group_identity)
        self.assertTrue(so3_group_product.group_element.shape == (3, 3))
        print(f'\nSO3 Product:{so3_group_product}')

        # Visualization
        LieSO3Group.visualize_all([so3_element_x, so3_point_identity], 0)
        so3_group_product.visualize('Composition of a Rotation along X\nwith identity', 0)

    @unittest.skip('Ignored')
    def test_product_3(self):
        # First SO3 rotation matrix
        so3_tangent_vec = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        so3_group = LieSO3Group.build(algebra_element=so3_tangent_vec)
        so3_point = LieElement(
            group_element=so3_group.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='3D Rotation:\n[0.4 0.3 0.8]\n[0.2 0.4 0.1]\n[0.1 0.2 0.6]')

        # Second SO3 rotation matrix
        so3_tangent_vec2 = [0.5]*len(so3_tangent_vec)
        so3_group2 = LieSO3Group.build(algebra_element=so3_tangent_vec2)
        so3_point2 = LieElement(
            group_element=so3_group2.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='3D Rotation:\n[0.5 0.5 0.5]\n[0.5 0.5 0.5]\n[0.5 0.5 0.5]')

        # Composition of two SO3 rotation matrices
        so3_group_product = so3_group.product(so3_group2)
        self.assertTrue(so3_group_product.group_element.shape == (3, 3))
        print(f'\nSO3 Product:{so3_group_product}')

        LieSO3Group.visualize_all([so3_point, so3_point2], 0)
        so3_group_product.visualize('Composition of 3D rotations')

    @unittest.skip('Ignored')
    def test_algebra(self):
        # Step 1: Define the unit generator of rotations along Y axis-- Ly
        ly_element = np.array([[0.0, 0.0, 1.0],
                               [0.0, 0.0, 0.0],
                               [-1.0, 0.0, 0.0]])

        # Step 2; Compute the element on SO3 manifold
        so3_group = LieSO3Group(algebra_element=ly_element)
        print(f'SO3 element: {so3_group}')

        # Step 3: Compute the algebra element from SO3 element
        lie_algebra = so3_group.lie_algebra()
        assert lie_algebra.size == len(ly_element)
        print(f'\nComputed algebra element:\n{lie_algebra}')


    @unittest.skip('Ignored')
    def test_algebra2(self):
        # First SO3 rotation matrix 90 degree along z axis
        lz_element = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # The SO3 element is computed at identity (to conform to Lie Algebra)
        so3_group = LieSO3Group.build(algebra_element=lz_element)
        print(f'SO3 point:\n{so3_group}')
        lie_algebra = so3_group.lie_algebra()
        print(f'\nLie algebra:\n{lie_algebra}')


    @unittest.skip('Ignored')
    def test_projection(self):
        # SO3 rotation along X axis
        lx_element = np.array([[0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0],
                               [0.0, 1.0, 0.0]])

        so3_group = LieSO3Group(algebra_element=lx_element)
        projected = so3_group.projection()
        print(f'\nProjected point with identity:\n{projected.group_element}')

        # Use a different reference point on SO3 manifold
        non_identity = np.array([[0.0, 0.0, 1.0],
                                 [0.0, 1.0, 0.0],
                                 [-1.0, 0.0, 0.0]])
        so3_group = LieSO3Group(algebra_element=lx_element, identity_element=non_identity)
        projected = so3_group.projection()
        print(f'\nProjected point reference\n{so3_group.algebra_element}:\n{projected.group_element}')

    @unittest.skip('Ignored')
    def test_bracket(self):
        # SO3 rotation along X axis
        lx_element = np.array([[0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0],
                               [0.0, 1.0, 0.0]])
        so3_group = LieSO3Group(algebra_element=lx_element)
        print(f'\nAlgebra element:\n{so3_group.algebra_element}')
        np.set_printoptions(precision=3)
        print(f'\nSO3 element\n{so3_group.group_element}')
        bracket = so3_group.bracket(lx_element)
        print(f'\nBracket [x,x]:\n{bracket}')


    # @unittest.skip('Ignored')
    def test_bracket2(self):
        # First SO3 rotation along X axis
        lx_element = np.array([[0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0],
                               [0.0, 1.0, 0.0]])
        so3_group_x = LieSO3Group(algebra_element=lx_element)
        print(f'\n{so3_group_x.group_element}')

        # Second SO3 rotation matrix along Y axis
        ly_element = np.array([[0.0, 0.0, 1.0],
                               [0.0, 0.0, 0.0],
                               [-1.0, 0.0, 0.0]])
        bracket = so3_group_x.bracket(ly_element)
        print(f'\nBracket:\n{bracket}')
        so3_group_y = LieSO3Group(algebra_element=ly_element)

        so3_element_x = LieElement(
            group_element= so3_group_x.algebra_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 element from rotation\n[0 0  0]\n[0 0 -1]\n[0 1  0]\n@ identity')
        print(f'SO3 element-x:\n{so3_element_x.group_element}')
        so3_element_y = LieElement(
            group_element= so3_group_y.algebra_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 element from rotation\n[ 0 0 1]\n[ 0 0 0]\n[-1 0  0]\n@ identity')
        print(f'SO3 element-y:\n{so3_element_y.group_element}')
        LieSO3Group.visualize_all([so3_element_x, so3_element_y], 0)

        so3_bracket_element = LieElement(
            group_element=bracket,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='Lie Bracket')
        LieSO3Group.visualize_all([so3_bracket_element], 0)


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

