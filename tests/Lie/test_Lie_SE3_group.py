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
        rot_matrix = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        trans_vector = [1.0, 3.0, 2.0]
        print(f'\nRotation matrix:\n{np.reshape(rot_matrix, (3, 3))}')
        print(f'Translation vector: {trans_vector}')
        lie_se3_group = LieSE3Group.build_from_vec(rot_matrix, trans_vector)
        print(lie_se3_group)

    @unittest.skip('Ignored')
    def test_inverse(self):
        rot_matrix = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        trans_vector = [0.5, 0.8, 0.6]
        print(f'\nRotation matrix:\n{np.reshape(rot_matrix, (3, 3))}')
        print(f'Translation vector: {trans_vector}')
        lie_se3_group = LieSE3Group.build_from_vec(rot_matrix, trans_vector)
        inv_lie_se3_group = lie_se3_group.inverse()
        print(f'\nSE3 element\n{lie_se3_group}\nInverse\n{inv_lie_se3_group}')

        import matplotlib.pyplot as plt
        plt.imshow(lie_se3_group.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'SE3 point\n{np.round(lie_se3_group.group_element, 2)}')
        plt.show()

        plt.imshow(inv_lie_se3_group.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'SE3 Inverse point\n{np.round(inv_lie_se3_group.group_element, 2)}')
        plt.show()

    @unittest.skip('Ignored')
    def test_product_1(self):
        # First SO3 rotation matrix 90 degree along x axis
        rot_matrix = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        trans_vector = [0.5, 0.8, 0.6]
        se3_group = LieSE3Group.build_from_vec(rot_matrix, trans_vector)

        # Composition of the same matrix
        so3_group_product = se3_group.product(se3_group)
        print(f'\nSO3 Product:{so3_group_product}')

        import matplotlib.pyplot as plt
        plt.imshow(so3_group_product.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        title = '[[1618.17    0.00    0.00  119568.07]\n[   0.00   40.51  -52.045     62.14]\n[   0.00   52.04   40.51     113.23]\n[   0.00    0.00    0.00    1618.17]]'
        plt.title(f'Composed SE3 point\n{title}')
        plt.show()


    def test_product_2(self):
        # First SO3 rotation matrix 90 degree along x axis
        rot1_matrix = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        trans1_vector = [0.5, 0.8, 0.6]
        se3_group1 = LieSE3Group.build_from_vec(rot1_matrix, trans1_vector)
        print(f'\nFirst SE3 point:{se3_group1}')

        rot2_matrix = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        trans2_vector = [0.1, -0.3, 0.3]
        se3_group2 = LieSE3Group.build_from_vec(rot2_matrix, trans2_vector)
        print(f'\nSecond SE3 point:{se3_group2}')

        # Composition of the same matrix
        se3_composed_group = se3_group1.product(se3_group1)
        print(f'\nComposed SE3 point:{se3_composed_group}')

        import matplotlib.pyplot as plt
        plt.imshow(se3_group1.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()

        plt.imshow(se3_group2.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()

        plt.imshow(se3_composed_group.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()



    @unittest.skip('Ignored')
    def test_visualize(self):
        rot_matrix = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        trans_vector = [0.5, 0.8, 0.6]
        print(f'\nRotation matrix:\n{np.reshape(rot_matrix, (3, 3))}')
        print(f'Translation vector: {trans_vector}')
        lie_se3_group = LieSE3Group.build_from_vec(rot_matrix, trans_vector)
        print(lie_se3_group)
        lie_se3_group.visualize_tangent_space(rot_matrix, trans_vector)

        import matplotlib.pyplot as plt
        plt.imshow(lie_se3_group.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'SE3 point\n{np.round(lie_se3_group.group_element, 2)}')
        plt.show()




