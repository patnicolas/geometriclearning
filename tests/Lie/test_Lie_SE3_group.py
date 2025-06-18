import unittest
import numpy as np
from Lie.SE3_visualization import SE3Visualization
from Lie.Lie_SE3_group import LieSE3Group
from Lie import u3d
import logging
import os
import python
from python import SKIP_REASON


class LieSE3GroupTest(unittest.TestCase):

    def test_build_from_numpy(self):
        np.set_printoptions(precision=3, suppress=True)

        rot_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        trans_matrix = np.array([1.0, 3.0, 2.0])
        lie_se3_group = LieSE3Group(rot_matrix, trans_matrix)
        logging.info(repr(lie_se3_group))
        self.assertTrue(lie_se3_group.se3_element.group_element.shape == (4, 4))

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_build_from_unit_elements(self):
        from Lie import u3d

        lie_se3_group = LieSE3Group(u3d.y_rot, u3d.x_trans)
        logging.info(lie_se3_group)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_inverse(self):
        lie_se3_group = LieSE3Group(rot_matrix=u3d.y_rot, trans_matrix=u3d.x_trans, epsilon=1e-4)
        tgt_vector = lie_se3_group.tangent_vector
        logging.info(tgt_vector)

        # Retrieve the inverse of the se3 group element
        inv_lie_se3_group = lie_se3_group.inverse()
        logging.info(f'\nSE3 element:\n{lie_se3_group}\nInverse:--\n{inv_lie_se3_group}')
        self.assertTrue(inv_lie_se3_group.this_group_element().shape == (4, 4))
        inv_tgt_vector = inv_lie_se3_group.tangent_vector
        logging.info(inv_tgt_vector)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_inverse_2(self):
        lie_se3_group = SE3Visualization(rot_matrix=u3d.y_rot, trans_matrix=u3d.x_trans)
        inv_lie_se3_group = lie_se3_group.inverse()
        logging.info(f'\nSE3 element\n{lie_se3_group}\nInverse\n{inv_lie_se3_group}')
        self.assertTrue(inv_lie_se3_group.belongs())

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(lie_se3_group.this_group_element(), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        np.set_printoptions(precision=3, suppress=False, floatmode='fixed')
        plt.title(f'SE3 Element\n{lie_se3_group.this_group_element()}')
        plt.show()

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(inv_lie_se3_group.this_group_element(), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'SE3 Inverse Element\n{inv_lie_se3_group.this_group_element()}')
        plt.show()

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_composition(self):
        first_trans = u3d.x_trans + u3d.y_trans + u3d.z_trans
        second_trans = u3d.x_trans - u3d.y_trans - 3
        se3_group_1 = SE3Visualization(rot_matrix=u3d.y_rot, trans_matrix=first_trans)
        se3_group_2 = SE3Visualization(rot_matrix=u3d.z_rot, trans_matrix=second_trans)

        # Composition
        se3_composed_group = se3_group_1.multiply(se3_group_2)
        self.assertTrue(se3_composed_group.belongs())
        logging.info(f'\nFirst element:\n{se3_group_1}\nSecond element\n{se3_group_2}'
                     f'\nComposed element: {se3_composed_group}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_self_composition(self):
        se3_group_1 = SE3Visualization(rot_matrix=u3d.x_rot, trans_matrix=u3d.y_trans)

        # Composition
        se3_composed_group = se3_group_1.multiply(se3_group_1)
        logging.info(f'\nFirst element:\n{se3_group_1}\nSelf Composed element: {se3_composed_group}')
        self.assertTrue(se3_composed_group.belongs())

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_se3_matrix_generation(self):
        rotation = np.array([[ 0.0, 1.0,  2.0],
                             [-1.0, 0.0, -0.5],
                             [-2.0, 0.5,  0.0]])
        translation = np.array([0.2, 0.6, -1.0])
        se3_matrix = np.eye(4)
        se3_matrix[:3, :3] = rotation
        se3_matrix[:3, 3] = translation.flatten()
        logging.info(se3_matrix)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_commutative(self):
        from Lie import u3d as u

        se3_1 = SE3Visualization(np.array(0.5 * u.x_rot + 2 * u.y_rot),
                                 np.array(u.y_trans + u.z_trans))
        se3_2 = SE3Visualization(np.array(u.y_rot - 3.0 * u.x_rot),
                                 np.array(u.x_trans))
        logging.info(f'\nse3_1 x se3_2 {se3_1.multiply(se3_2)}')
        logging.info(f'\nse3_2 x se3_1 {se3_2.multiply(se3_1)}')
        self.assertFalse(se3_1.multiply(se3_2) == se3_2.multiply(se3_1))

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_matrix_generation(self):
        R = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        t = np.array([[1.0, 2.0, 3.0]])

        se3_matrix = np.eye(4)
        se3_matrix[:3, :3] = R
        se3_matrix[:3, 3] = t.squeeze(axis=0).flatten()

        rot, trans = LieSE3Group.reshape(R, t)
        algebra_element = np.concatenate([rot, trans], axis=1)

        self.assertTrue(se3_matrix.all() == algebra_element.all())
        logging.info(se3_matrix)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_visualize(self):
        # Two inputs= in the tangent space: 3x3 90 degree rotation along Y axis
        # and Translation along X xis
        logging.info(f'\nRotation matrix:\n{np.reshape(u3d.y_rot, (3, 3))}')
        lie_se3_group = SE3Visualization(u3d.y_rot, u3d.x_trans)
        logging.info(lie_se3_group)
        lie_se3_group.visualize_tangent_space(u3d.y_rot, u3d.x_trans)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        fig.set_facecolor('#F2F9FE')
        plt.imshow(lie_se3_group.se3_element.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        np.set_printoptions(precision=3, suppress=False, floatmode='fixed')
        plt.title(f'SE3 element\n{lie_se3_group.se3_element.group_element}')
        plt.show()
