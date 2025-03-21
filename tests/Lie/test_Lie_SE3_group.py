import unittest
import numpy as np
from Lie.Lie_SE3_group import LieSE3Group


class LieSE3GroupTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_build_from_numpy(self):
        rot_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        trans_matrix = np.array([[1.0, 3.0, 2.0]])
        lie_se3_group = LieSE3Group.build_from_numpy(rot_matrix, trans_matrix)
        print(lie_se3_group)

    @unittest.skip('Ignored')
    def test_build_from_vec(self):
        # Two inputs
        # - Unit generator of rotations along Y axis
        # - Generator for a translation along X xis
        ly_rotation = np.array([[0.0, 0.0, 1.0],
                                [0.0, 0.0, 0.0],
                                [-1.0, 0.0, 0.0]])
        x_translation = np.array([[1.0, 0.0, 0.0]])
        print(f'\nRotation matrix:\n{np.reshape(ly_rotation, (3, 3))}')
        print(f'Translation vector: {x_translation}')
        lie_se3_group = LieSE3Group(ly_rotation, x_translation)
        print(lie_se3_group)

    @unittest.skip('Ignored')
    def test_inverse(self):
        ly_rotation = np.array([[0.0, 0.0, 1.0],
                                [0.0, 0.0, 0.0],
                                [-1.0, 0.0, 0.0]])
        x_translation = np.array([[1.0, 0.0, 0.0]])
        lie_se3_group = LieSE3Group(ly_rotation, x_translation)
        inv_lie_se3_group = lie_se3_group.inverse()
        print(f'\nSE3 element\n{lie_se3_group}\nInverse\n{inv_lie_se3_group}')

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(lie_se3_group.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        np.set_printoptions(precision=3, suppress=False, floatmode='fixed')
        plt.title(f'SE3 Element\n{lie_se3_group.group_element}')
        plt.show()

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(inv_lie_se3_group.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'SE3 Inverse Element\n{inv_lie_se3_group.group_element}')
        plt.show()

    @unittest.skip('Ignored')
    def test_product_1(self):
        ly_rotation = np.array([[0.0, 0.0, 1.0],
                                [0.0, 0.0, 0.0],
                                [-1.0, 0.0, 0.0]])
        x_translation = np.array([[1.0, 0.0, 0.0]])
        se3_group = LieSE3Group(ly_rotation, x_translation)

        # Composition of the same SE3 element
        so3_group_product = se3_group.product(se3_group)
        print(f'\nSelf composed SE3 elements:\n{so3_group_product.group_element}')

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(so3_group_product.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        np.set_printoptions(precision=3, suppress=False, floatmode='fixed')
        plt.title(f'Self composed SE3 element\n{so3_group_product.group_element}')
        plt.show()

    @unittest.skip('Ignored')
    def test_product_2(self):
        ly_rotation = np.array([[0.0, 0.0, 1.0],
                                [0.0, 0.0, 0.0],
                                [-1.0, 0.0, 0.0]])
        lz_rotation = np.array([[0.0, -1.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]])
        y_translation = np.array([[0.0, 1.0, 0.0]])
        z_translation = np.array([[0.0, 0.0, 1.0]])

        # First SO3 rotation matrix 90 degree along x axis
        se3_group_y = LieSE3Group(ly_rotation, y_translation)
        print(f'\nFirst SE3 element:{se3_group_y}')
        se3_group_z = LieSE3Group(lz_rotation, z_translation)
        print(f'\nSecond SE3 element:{se3_group_z}')

        # Composition of the same matrix
        se3_composed_group = se3_group_y.product(se3_group_z)
        print(f'\nComposed SE3 elements:{se3_composed_group}')

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        np.set_printoptions(precision=3, suppress=False, floatmode='fixed')
        plt.imshow(se3_group_y.group_element, cmap='viridis', interpolation='nearest')
        plt.title(f'First SE3 element\n{se3_group_y.group_element}')
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


    # @unittest.skip('Ignored')
    def test_visualize(self):
        # Two inputs= in the tangent space: 3x3 90 degree rotation along Y axis
        # and Translation along X xis
        ly_rotation = np.array([[0.0, 0.0, 1.0],
                                [0.0, 0.0, 0.0],
                                [-1.0, 0.0, 0.0]])
        x_translation = np.array([[1.0, 0.0, 0.0]])
        print(f'\nRotation matrix:\n{np.reshape(ly_rotation, (3, 3))}')
        print(f'Translation vector: {x_translation}')
        lie_se3_group = LieSE3Group(ly_rotation, x_translation)
        print(lie_se3_group)
        lie_se3_group.visualize_tangent_space(ly_rotation, x_translation)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        fig.set_facecolor('#F2F9FE')
        plt.imshow(lie_se3_group.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        np.set_printoptions(precision=3, suppress=False, floatmode='fixed')
        plt.title(f'SE3 element\n{lie_se3_group.group_element}')
        plt.show()




