import unittest
import numpy as np
from Lie.Lie_SE3_group import LieSE3Group, VisualTangentVector
from Lie import UnitElements
from typing import List
import geomstats.backend as gs


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

    @unittest.skip('Ignored')
    def test_build_from_unit_elements(self):
        # Two inputs
        # - Unit generator of rotations along Y axis
        # - Generator for a translation along X xis

        epsilon = 1e-4
        print(f'\nRotation matrix:\n{np.reshape(UnitElements.y_rot, (3, 3))}')
        print(f'Translation vector: {UnitElements.x_trans}')
        lie_se3_group = LieSE3Group(UnitElements.y_rot, UnitElements.x_trans, epsilon, point_type='vector')
        print(lie_se3_group)

        rot_trans: List[float] = [1.9, 0.1, 0.35, 1.5, 1.5, 0.5]
        rot_trans_str = ', '.join([str(x) for x in rot_trans])
        visual_tangent_vectors_1 = VisualTangentVector(vec=gs.array(rot_trans),
                                                       x=0.0,
                                                       y=-4.1,
                                                       z=-4.0,
                                                       s=f'Vector [{rot_trans_str}]',
                                                       color='red')
        rot_trans: List[float] = [-1.6, 0.5, 2.5, -1.9, 2.2, 2.3]
        rot_trans_str = ', '.join([str(x) for x in rot_trans])
        visual_tangent_vectors_2 = VisualTangentVector(vec=gs.array(rot_trans),
                                                       x=-0.1,
                                                       y=-4.0,
                                                       z=-4.8,
                                                       s=f'Vector [{rot_trans_str}]',
                                                       color='blue')
        lie_se3_group.visualize_displacements(visual_tangent_vecs=[visual_tangent_vectors_1, visual_tangent_vectors_2],
                                              num_points=84,
                                              title='84 Displacements Points - two SE(3) @ Identity')

    @unittest.skip('Ignored')
    def test_inverse(self):
        lie_se3_group = LieSE3Group(rot_matrix=UnitElements.y_rot,
                                    trans_matrix=UnitElements.x_trans,
                                    epsilon=1e-4)
        tgt_vector = lie_se3_group.tangent_vector

        # Retrieve the inverse of the se3 group element
        inv_lie_se3_group = lie_se3_group.inverse()
        print(f'\nSE3 element:\n{lie_se3_group}\nInverse:--\n{inv_lie_se3_group}')
        self.assertTrue(inv_lie_se3_group.this_group_element().shape == (4, 4))
        inv_tgt_vector = inv_lie_se3_group.tangent_vector

        vec_descriptor = ("       Original Algebra\n 0.000 0.000 1.000 1.000\n 0.000 0.000 0.000 0.000"
                          "\n-1.000 0.000 0.000 0.000\n 0.000 0.000 0.000 1.000")
        visual_tangent_vectors = VisualTangentVector(vec=tgt_vector,
                                                     x=-2.4,
                                                     y=-2.1,
                                                     z=3.1,
                                                     s=vec_descriptor,
                                                     color='red')

        inv_vec_descriptor = ("       Inverse Algebra\n 0.540 0.000 0.841  1.510\n 0.000 1.000 0.000  0.000"
                              "\n-0.841 0.000 0.540 -0.668\n 0.000 0.000 0.000  2.718")
        visual_inv_tangent_vectors = VisualTangentVector(vec=inv_tgt_vector,
                                                         x=1.0,
                                                         y=0.1,
                                                         z=3.1,
                                                         s=inv_vec_descriptor,
                                                         color='blue')

        se3_test_group = LieSE3Group(rot_matrix=UnitElements.y_rot,
                                     trans_matrix=UnitElements.x_trans,
                                     epsilon=1e-4,
                                     point_type='vector')
        se3_test_group.visualize_displacements(visual_tangent_vecs=[visual_tangent_vectors, visual_inv_tangent_vectors],
                                               num_points=48,
                                               scale=(-1.5, 1.5),
                                               title='48 Displacement Points - SE(3) Inverse @ Identity')

    @unittest.skip('Ignored')
    def test_inverse_2(self):
        lie_se3_group = LieSE3Group(rot_matrix=UnitElements.y_rot,
                                    trans_matrix=UnitElements.x_trans)
        inv_lie_se3_group = lie_se3_group.inverse()
        print(f'\nSE3 element\n{lie_se3_group}\nInverse\n{inv_lie_se3_group}')
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

    @unittest.skip('Ignored')
    def test_composition(self):
        first_trans = UnitElements.x_trans + UnitElements.y_trans + UnitElements.z_trans
        second_trans = UnitElements.x_trans - UnitElements.y_trans - 3
        se3_group_1 = LieSE3Group(rot_matrix=UnitElements.y_rot, trans_matrix=first_trans)
        se3_group_2 = LieSE3Group(rot_matrix=UnitElements.z_rot, trans_matrix=second_trans)

        # Composition
        se3_composed_group = se3_group_1.multiply(se3_group_2)
        self.assertTrue(se3_composed_group.belongs())
        print(f'\nFirst element:\n{se3_group_1}\nSecond element\n{se3_group_2}'
              f'\nComposed element: {se3_composed_group}')

        vec_descriptor_1 = ("          First Algebra\n 0.000 0.000 1.000 1.000\n 0.000 0.000 0.000 1.000"
                            "\n-1.000 0.000 0.000 1.000\n 0.000 0.000 0.000 1.000  ")
        visual_tangent_vectors_1 = VisualTangentVector(vec=se3_group_1.tangent_vector,
                                                       x=-4.4,
                                                       y=-4.5,
                                                       z=-0.6,
                                                       s=vec_descriptor_1,
                                                       color='red')

        vec_descriptor_2 = ("         Second Algebra\n0.000 -1.000 0.000 -2.000\n1.000  0.000 0.000 -4.000"
                            "\n0.000  0.000 0.000 -3.000\n0.000  0.000 0.000  1.000 ")
        visual_tangent_vectors_2 = VisualTangentVector(vec=se3_group_2.tangent_vector,
                                                       x=-4.3,
                                                       y=-4.5,
                                                       z=2.2,
                                                       s=vec_descriptor_2,
                                                       color='blue')

        composed_vec_descriptor = ("       Composed Algebra\n 0.292 -0.455 0.841  1.396\n 0.841  0.540 0.000 -2.705"
                                   "\n-0.455  0.708 0.540 -0.206\n 0.000  0.000 0.000  1.000")
        visual_composed_tangent_vector = VisualTangentVector(vec=se3_composed_group.tangent_vector,
                                                             x=-4.2,
                                                             y=-4.5,
                                                             z=5.0,
                                                             s=composed_vec_descriptor,
                                                             color='green')

        se3_test_group = LieSE3Group(rot_matrix=UnitElements.y_rot,
                                     trans_matrix=UnitElements.x_trans,
                                     epsilon=1e-4,
                                     point_type='vector')
        se3_test_group.visualize_displacements(visual_tangent_vecs=[visual_tangent_vectors_1,
                                                                    visual_tangent_vectors_2,
                                                                    visual_composed_tangent_vector
                                                                    ],
                                               num_points=64,
                                               scale=(-0.7, 0.7),
                                               title='64 Displacement Points - SE(3) Composition @ Identity')

    @unittest.skip('Ignored')
    def test_self_composition(self):
        se3_group_1 = LieSE3Group(rot_matrix=UnitElements.x_rot, trans_matrix=UnitElements.y_trans)

        # Composition
        se3_composed_group = se3_group_1.multiply(se3_group_1)
        print(f'\nFirst element:\n{se3_group_1}\nSelf Composed element: {se3_composed_group}')
        self.assertTrue(se3_composed_group.belongs())

        vec_descriptor_1 = ("              Algebra\n0.000 0.000  0.000 0.000\n0.000 0.000 -1.000 1.000"
                            "\n0.000 1.000  0.000 0.000\n0.000 0.000  0.000 1.000")
        visual_tangent_vectors_1 = VisualTangentVector(vec=se3_group_1.tangent_vector,
                                                       x=-3.1,
                                                       y=-1.1,
                                                       z=-0.4,
                                                       s=vec_descriptor_1,
                                                       color='red')

        composed_vec_descriptor = (" Self Composed Algebra\n1.000  0.000  0.000 0.000\n0.000 -0.416 -0.909 4.357"
                                   "\n0.000  0.989 -0.416 2.448\n0.000  0.000  0.000 1.000")
        visual_composed_tangent_vector = VisualTangentVector(vec=se3_composed_group.tangent_vector,
                                                             x=-3.0,
                                                             y=-1.1,
                                                             z=1.3,
                                                             s=composed_vec_descriptor,
                                                             color='blue')
        se3_test_group = LieSE3Group(rot_matrix=UnitElements.y_rot,
                                     trans_matrix=UnitElements.x_trans,
                                     epsilon=1e-4,
                                     point_type='vector')
        se3_test_group.visualize_displacements(visual_tangent_vecs=[visual_tangent_vectors_1,
                                                                    visual_composed_tangent_vector
                                                                    ],
                                               num_points=48,
                                               scale=(-0.35, 0.35),
                                               title='48 Displacement Points - SE(3) Self Composition @ Identity')

    def test_animate_displacements(self):
        epsilon = 1e-4
        print(f'\nRotation matrix:\n{np.reshape(UnitElements.y_rot, (3, 3))}')
        print(f'Translation vector: {UnitElements.x_trans}')
        lie_se3_group = LieSE3Group(UnitElements.y_rot, UnitElements.x_trans, epsilon, point_type='vector')
        print(lie_se3_group)

        rot_trans: List[float] = [1.9, 0.1, 0.35, 1.5, 1.5, 0.5]
        rot_trans_str = ', '.join([str(x) for x in rot_trans])
        visual_tangent_vectors_1 = VisualTangentVector(vec=gs.array(rot_trans),
                                                       x=0.0,
                                                       y=-4.1,
                                                       z=-4.0,
                                                       s=f'Vector [{rot_trans_str}]',
                                                       color='red')
        rot_trans: List[float] = [-1.6, 0.5, 2.5, -1.9, 2.2, 2.3]
        rot_trans_str = ', '.join([str(x) for x in rot_trans])
        visual_tangent_vectors_2 = VisualTangentVector(vec=gs.array(rot_trans),
                                                       x=-0.1,
                                                       y=-4.0,
                                                       z=-4.8,
                                                       s=f'Vector [{rot_trans_str}]',
                                                       color='blue')
        lie_se3_group.animate_displacements(visual_tangent_vecs=[visual_tangent_vectors_2],
                                            num_points=84,
                                            title='84 Displacements Points - SE(3) @ Identity')

    @unittest.skip('Ignored')
    def test_se3_matrix_generation(self):
        rotation = np.array([[ 0.0, 1.0,  2.0],
                             [-1.0, 0.0, -0.5],
                             [-2.0, 0.5,  0.0]])
        translation = np.array([0.2, 0.6, -1.0])
        se3_matrix = np.eye(4)
        se3_matrix[:3, :3] = rotation
        se3_matrix[:3, 3] = translation.flatten()
        print(se3_matrix)

    @unittest.skip('Ignored')
    def test_commutative(self):
        from Lie import unit_elements as u

        se3_1 = LieSE3Group(np.array(0.5*u.x_rot + 2*u.y_rot),
                            np.array(u.y_trans + u.z_trans))
        se3_2 = LieSE3Group(np.array(u.y_rot - 3.0*u.x_rot),
                            np.array(u.x_trans))
        print(f'\nse3_1 x se3_2 {se3_1.multiply(se3_2)}')
        print(f'\nse3_2 x se3_1 {se3_2.multiply(se3_1)}')
        self.assertFalse(se3_1.multiply(se3_2) == se3_2.multiply(se3_1))


    @unittest.skip('Ignored')
    def test_visualize(self):
        # Two inputs= in the tangent space: 3x3 90 degree rotation along Y axis
        # and Translation along X xis
        print(f'\nRotation matrix:\n{np.reshape(UnitElements.y_rot, (3, 3))}')
        lie_se3_group = LieSE3Group(UnitElements.y_rot, UnitElements.x_trans)
        print(lie_se3_group)
        lie_se3_group.visualize_tangent_space(UnitElements.y_rot, UnitElements.x_trans)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        fig.set_facecolor('#F2F9FE')
        plt.imshow(lie_se3_group.se3_element.group_element, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        np.set_printoptions(precision=3, suppress=False, floatmode='fixed')
        plt.title(f'SE3 element\n{lie_se3_group.se3_element.group_element}')
        plt.show()
