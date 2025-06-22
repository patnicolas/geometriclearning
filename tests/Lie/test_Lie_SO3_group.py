import unittest

from lie.lie_so3_group import LieSO3Group
from lie.lie_so3_group import LieElement
import numpy as np
from typing import AnyStr, List
from lie import u3d
import logging
import os
import python
from python import SKIP_REASON


class LieSO3GroupTest(unittest.TestCase):
    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init(self):
        so3_element = np.array([[0.0, 0.0, 0.0],
                                [0.0, -1.0, -1.0],
                                [0.0, 1.0, -1.0]])

        det = np.linalg.trace(so3_element)
        logging.info(det)
        so3_group = LieSO3Group(so3_element)
        logging.info(str(so3_group))
        self.assertTrue(so3_group.group_element.shape == (3, 3))

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_2(self):
        so3_group = LieSO3Group(u3d.y_rot)
        so3_element_1 = LieElement(
            group_element=so3_group.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 Ly element\n[0 0  1]\n[0 0 0]\n[-1 0  0]\n@ identity element')

        self.assertTrue(so3_group.group_element.shape == (3, 3))

        so3_group2 = LieSO3Group(u3d.z_rot)
        so3_element_2 = LieElement(
            group_element=so3_group2.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 Lz element\nIdentity element:\n[0 -1  0]\n[1  0  0]\n[0  0  0]')
        logging.info(str(so3_group2))
        LieSO3Group.visualize_all(so3_elements=[so3_element_1, so3_element_2], notation_index=3)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_3(self):
        cx = 2.0
        cy = -0.5
        cz = 1.5

        element = cx * u3d.x_rot + cy * u3d.y_rot + cz * u3d.z_rot
        logging.info(element)
        so3_group = LieSO3Group(element)
        so3_element = LieElement(
            group_element=so3_group.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 element for\n[0.0 -1.5 -0.5]\n[1.5  0.0 -2.0]\n[0.5  2.0  0.0]\n@ identity element')
        LieSO3Group.visualize_all(so3_elements=[so3_element], notation_index=3)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_inverse(self):
        # Generator for rotation along Y axis
        so3_group = LieSO3Group(u3d.y_rot)
        element = LieElement(
            group_element=so3_group.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 point from tangent vector\n[1 0  0]\n[0 0 -1]\n[0 1  0]\nBase point: Identity')

        logging.info(str(so3_group))

        # Inverse SO3 rotation matrix
        so3_inv_group = so3_group.inverse()
        inv_so3_inverse = LieElement(
            group_element=so3_inv_group.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 inverse point')
        logging.info(f'SO3 Inverse point:{so3_inv_group}')
        self.assertTrue(so3_inv_group.group_element.shape == (3, 3))
        # Visualization
        LieSO3Group.visualize_all([element, inv_so3_inverse], 2)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_product_1(self):
        # First SO3 rotation matrix along X axis
        so3_group_x = LieSO3Group(algebra_element=u3d.x_rot)
        so3_element_x = LieElement(
            group_element=so3_group_x.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 Rotation along X\n[0 0  0]\n[0 0 -1]\n[0 1  0]\n@ identity')

        # Second SO3 rotation along Y axis
        so3_group_y = LieSO3Group(algebra_element=u3d.y_rot)
        so3_point_y = LieElement(
            group_element=so3_group_y.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 Rotation along Y\n[ 0 0 1]\n[ 0 0 0]\n[-1 0 0]\n@ identity')

        # Composition of two SO3 rotation matrices
        so3_group_product = so3_group_x.product(so3_group_y)
        self.assertTrue(so3_group_product.group_element.shape == (3, 3))
        logging.info(f'\nSO3 Product:{so3_group_product}')

        # Visualization
        LieSO3Group.visualize_all([so3_element_x, so3_point_y], 0)
        so3_group_product.visualize('Composition of two SO3 matrices\nRotation along X with Y axis', 0)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_product_2(self):
        # First SO3 rotation matrix along X axis
        so3_group_x = LieSO3Group(algebra_element=u3d.x_rot)
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
        logging.info(f'\nSO3 Product:{so3_group_product}')

        # Visualization
        LieSO3Group.visualize_all([so3_element_x, so3_point_identity], 0)
        so3_group_product.visualize('Composition of a Rotation along X\nwith identity', 0)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_product_3(self):
        # First SO3 rotation matrix
        so3_tangent_vec = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        so3_group = LieSO3Group.build_identity(algebra_element=so3_tangent_vec)
        so3_point = LieElement(
            group_element=so3_group.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='3D Rotation:\n[0.4 0.3 0.8]\n[0.2 0.4 0.1]\n[0.1 0.2 0.6]')

        # Second SO3 rotation matrix
        so3_tangent_vec2 = [0.5]*len(so3_tangent_vec)
        so3_group2 = LieSO3Group.build_identity(algebra_element=so3_tangent_vec2)
        so3_point2 = LieElement(
            group_element=so3_group2.group_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='3D Rotation:\n[0.5 0.5 0.5]\n[0.5 0.5 0.5]\n[0.5 0.5 0.5]')

        # Composition of two SO3 rotation matrices
        so3_group_product = so3_group.product(so3_group2)
        self.assertTrue(so3_group_product.group_element.shape == (3, 3))
        logging.info(f'\nSO3 Product:{so3_group_product}')

        LieSO3Group.visualize_all([so3_point, so3_point2], 0)
        so3_group_product.visualize('Composition of 3D rotations')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_algebra(self):
        # Step 2; Compute the element on SO3 manifold
        so3_group = LieSO3Group(algebra_element=u3d.y_rot)
        logging.info(f'SO3 element: {so3_group}')

        # Step 3: Compute the algebra element from SO3 element
        lie_algebra = so3_group.lie_algebra()
        assert lie_algebra.size == len(u3d.y_rot)
        logging.info(f'\nComputed algebra element:\n{lie_algebra}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_algebra2(self):
        # The SO3 element is computed at identity (to conform to lie Algebra)
        so3_group = LieSO3Group(algebra_element=u3d.z_rot)
        logging.info(f'SO3 point:\n{so3_group}')
        lie_algebra = so3_group.lie_algebra()
        logging.info(f'\nlie algebra:\n{lie_algebra}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_projection(self):
        # SO3 rotation along X axis
        so3_group = LieSO3Group(algebra_element=u3d.x_rot)
        projected = so3_group.projection()
        logging.info(f'\nProjected point with identity:\n{projected.group_element}')

        # Use a different reference point on SO3 manifold
        non_identity = np.array([[0.0, 0.0, 1.0],
                                 [0.0, 1.0, 0.0],
                                 [-1.0, 0.0, 0.0]])
        so3_group = LieSO3Group(algebra_element=u3d.x_rot, identity_element=non_identity)
        projected = so3_group.projection()
        logging.info(f'\nProjected point reference\n{so3_group.algebra_element}:\n{projected.group_element}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_bracket(self):
        # SO3 rotation along X axis
        so3_group = LieSO3Group(algebra_element=u3d.x_rot)
        logging.info(f'\nAlgebra element:\n{so3_group.algebra_element}')
        np.set_printoptions(precision=3)
        logging.info(f'\nSO3 element\n{so3_group.group_element}')
        bracket = so3_group.bracket(u3d.x_rot)
        logging.info(f'\nBracket [x,x]:\n{bracket}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_bracket2(self):
        # First SO3 rotation along X axis
        so3_group_x = LieSO3Group(algebra_element=u3d.x_rot)
        logging.info(f'\n{so3_group_x.group_element}')

        # Second SO3 rotation matrix along Y axis
        bracket = so3_group_x.bracket(u3d.y_rot)
        logging.info(f'\nBracket:\n{bracket}')
        so3_group_y = LieSO3Group(algebra_element=u3d.y_rot)

        so3_element_x = LieElement(
            group_element= so3_group_x.algebra_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 element from rotation\n[0 0  0]\n[0 0 -1]\n[0 1  0]\n@ identity')
        logging.info(f'SO3 element-x:\n{so3_element_x.group_element}')
        so3_element_y = LieElement(
            group_element= so3_group_y.algebra_element,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='SO3 element from rotation\n[ 0 0 1]\n[ 0 0 0]\n[-1 0  0]\n@ identity')
        logging.info(f'SO3 element-y:\n{so3_element_y.group_element}')
        LieSO3Group.visualize_all([so3_element_x, so3_element_y], 0)

        so3_bracket_element = LieElement(
            group_element=bracket,
            identity_element=LieSO3Group.identity_matrix,
            descriptor='lie Bracket')
        LieSO3Group.visualize_all([so3_bracket_element], 0)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
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

