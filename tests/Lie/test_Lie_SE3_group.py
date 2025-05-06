import unittest
import numpy as np
from Lie.Lie_SE3_group import LieSE3Group, SE3UnitElements

class LieSE3GroupTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_build_from_numpy(self):
        np.set_printoptions(precision=3, suppress=True)

        rot_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        trans_matrix = np.array([[1.0, 3.0, 2.0]])
        epsilon = 1e-4
        lie_se3_group = LieSE3Group(rot_matrix, trans_matrix, epsilon)
        print(repr(lie_se3_group))
        self.assertTrue(lie_se3_group.se3_element.group_element.shape == (4, 4))

    # @unittest.skip('Ignored')
    def test_build_from_unit_elements(self):
        # Two inputs
        # - Unit generator of rotations along Y axis
        # - Generator for a translation along X xis

        epsilon = 1e-4
        print(f'\nRotation matrix:\n{np.reshape(SE3UnitElements.y_rot, (3, 3))}')
        print(f'Translation vector: {SE3UnitElements.x_trans}')
        lie_se3_group = LieSE3Group(SE3UnitElements.y_rot, SE3UnitElements.x_trans, epsilon, point_type='vector')
        print(lie_se3_group)

        import geomstats.backend as gs
        lie_se3_group.visualize_geodesics(gs.array([1.8, 0.2, 0.3, 3.0, 3.0, 1.0]))

        lie_se3_group = LieSE3Group(SE3UnitElements.x_rot, SE3UnitElements.y_trans, epsilon, point_type='vector')
        lie_se3_group.visualize_geodesics(gs.array([0.8, 1.2, 2.3, 0.0, 2.0, 6.0]))



    @unittest.skip('Ignored')
    def test_inverse(self):
        lie_se3_group = LieSE3Group(rot_matrix=SE3UnitElements.y_rot,
                                    trans_matrix=SE3UnitElements.x_trans,
                                    epsilon=1e-4)
        inv_lie_se3_group = lie_se3_group.inverse()
        print(f'\nSE3 element\n{lie_se3_group}\nInverse\n{inv_lie_se3_group}')
        self.assertTrue(inv_lie_se3_group.this_group_element().shape == (4, 4))

    @unittest.skip('Ignored')
    def test_inverse_2(self):
        lie_se3_group = LieSE3Group(rot_matrix=SE3UnitElements.y_rot,
                                    trans_matrix=SE3UnitElements.x_trans)
        inv_lie_se3_group = lie_se3_group.inverse()
        print(f'\nSE3 element\n{lie_se3_group}\nInverse\n{inv_lie_se3_group}')

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

    @unittest.skip('Ignored')
    def test_product_1(self):
        se3_group = LieSE3Group(SE3UnitElements.y_rot, SE3UnitElements.x_trans)

        # Composition of the same SE3 element
        so3_group_product = se3_group.compose(se3_group)
        print(f'\nSelf composed SE3 elements:\n{so3_group_product.this_group_element()}')

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(so3_group_product.this_group_element(), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        np.set_printoptions(precision=3, suppress=False, floatmode='fixed')
        plt.title(f'Self composed SE3 element\n{so3_group_product.this_group_element()}')
        plt.show()

    @unittest.skip('Ignored')
    def test_product_2(self):
        # First SO3 rotation matrix 90 degree along x axis
        se3_group_y = LieSE3Group(SE3UnitElements.y_rot, SE3UnitElements.y_trans)
        print(f'\nFirst SE3 element:{se3_group_y}')
        se3_group_z = LieSE3Group(SE3UnitElements.z_rot, SE3UnitElements.z_trans)
        print(f'\nSecond SE3 element:{se3_group_z}')

        # Composition of the same matrix
        se3_composed_group = se3_group_y.compose(se3_group_z)
        print(f'\nComposed SE3 elements:{se3_composed_group}')

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        np.set_printoptions(precision=3, suppress=False, floatmode='fixed')
        plt.imshow(se3_group_y.this_group_element(), cmap='viridis', interpolation='nearest')
        plt.title(f'First SE3 element\n{se3_group_y.this_group_element()}')
        plt.colorbar()
        plt.show()

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(se3_group_z.group_element, cmap='viridis', interpolation='nearest')
        plt.title(f'Second SE3 element\n{se3_group_z.group_element}')
        plt.colorbar()
        plt.show()

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(se3_composed_group.group_element, cmap='viridis', interpolation='nearest')
        plt.title(f'Composed SE3 element\n{se3_group_z.group_element}')
        plt.colorbar()
        plt.show()

    @unittest.skip('Ignored')
    def test_visualize(self):
        # Two inputs= in the tangent space: 3x3 90 degree rotation along Y axis
        # and Translation along X xis
        print(f'\nRotation matrix:\n{np.reshape(SE3UnitElements.y_rot, (3, 3))}')
        lie_se3_group = LieSE3Group(SE3UnitElements.y_rot, SE3UnitElements.x_trans)
        print(lie_se3_group)
        lie_se3_group.visualize_tangent_space(SE3UnitElements.y_rot, SE3UnitElements.x_trans)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        fig.set_facecolor('#F2F9FE')
        plt.imshow(lie_se3_group.se3_element.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        np.set_printoptions(precision=3, suppress=False, floatmode='fixed')
        plt.title(f'SE3 element\n{lie_se3_group.se3_element.group_element}')
        plt.show()
