
import unittest
from geometry.lie import SE3ElementDescriptor
from geometry.lie import u3d
from geometry.lie.se3_visualization import SE3Visualization
import numpy as np
from typing import List
import geomstats.backend as gs
import logging
import os
from python import SKIP_REASON



class SE3VisualizationTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_animation_one(self):
        from geometry.lie import u3d

        visualization = SE3Visualization(u3d.y_rot, u3d.x_trans)
        logging.info(visualization)

        rot_trans: List[float] = [-1.6, 0.5, 2.5, -1.9, 2.2, 2.3]
        rot_trans_str = ', '.join([str(x) for x in rot_trans])
        visual_tangent_vectors = SE3ElementDescriptor(vec=gs.array(rot_trans),
                                                      x=-0.1,
                                                      y=-4.0,
                                                      z=-4.8,
                                                      s=f'Vector [{rot_trans_str}]',
                                                      color='red')
        num_points = 96
        visualization.animate(se3_element_descs=[visual_tangent_vectors],
                              num_points=num_points,
                              initial_point=visualization.identity(),
                              scale=(-1.0, 1.0),
                              title=f'{num_points} Displacements Points Animation - SE(3)',
                              interval=1000,
                              fps=40)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_animation_two(self):
        from geometry.lie import u3d

        visualization = SE3Visualization(u3d.y_rot, u3d.x_trans)
        logging.info(visualization)

        rot_trans: List[float] = [1.9, 0.1, 0.35, 1.5, 1.5, 0.5]
        rot_trans_str = ', '.join([str(x) for x in rot_trans])
        visual_tangent_vectors_1 = SE3ElementDescriptor(vec=gs.array(rot_trans),
                                                        x=0.0,
                                                        y=-4.1,
                                                        z=-4.0,
                                                        s=f'Vector [{rot_trans_str}]',
                                                        color='red')
        rot_trans: List[float] = [-1.6, 0.5, 2.5, -1.9, 2.2, 2.3]
        rot_trans_str = ', '.join([str(x) for x in rot_trans])
        visual_tangent_vectors_2 = SE3ElementDescriptor(vec=gs.array(rot_trans),
                                                        x=-0.1,
                                                        y=-4.0,
                                                        z=-4.8,
                                                        s=f'Vector [{rot_trans_str}]',
                                                        color='blue')
        num_points = 128
        visualization.animate(se3_element_descs=[visual_tangent_vectors_1, visual_tangent_vectors_2],
                              num_points=num_points,
                              initial_point=visualization.identity(),
                              scale=(-1.0, 1.0),
                              title=f'{num_points} Displacements Points Animation - SE(3)',
                              interval=400,
                              fps=40)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_visualization_2(self):
        from geometry.lie import u3d

        logging.info(f'\nRotation matrix:\n{np.reshape(u3d.y_rot, (3, 3))}')
        logging.info(f'Translation vector: {u3d.x_trans}')
        se3_visualization = SE3Visualization(u3d.y_rot, u3d.x_trans)
        logging.info(se3_visualization)

        # First SE3 element
        rot_trans: List[float] = [1.9, 0.1, 0.35, 1.5, 1.5, 0.5]
        rot_trans_str = ', '.join([str(x) for x in rot_trans])
        se3_element_1_desc = SE3ElementDescriptor(vec=gs.array(rot_trans),
                                                  x=0.0,
                                                  y=-4.1,
                                                  z=-4.0,
                                                  s=f'Vector [{rot_trans_str}]',
                                                  color='red')

        # Second SE3 element
        rot_trans: List[float] = [-1.6, 0.5, 2.5, -1.9, 2.2, 2.3]
        rot_trans_str = ', '.join([str(x) for x in rot_trans])
        se3_element_2_desc = SE3ElementDescriptor(vec=gs.array(rot_trans),
                                                  x=-0.1,
                                                  y=-4.0,
                                                  z=-4.8,
                                                  s=f'Vector [{rot_trans_str}]',
                                                  color='blue')
        se3_visualization.visualize(se3_element_descs=[se3_element_1_desc, se3_element_2_desc],
                                    initial_point=se3_visualization.identity(),
                                    scale=(-1.0, 1.0),
                                    num_points=84,
                                    title='84 Displacements Points - two SE(3) @ Identity')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_inverse_visualization(self):
        se3_lie_group = SE3Visualization(rot_matrix=u3d.y_rot, trans_matrix=u3d.x_trans)

        tgt_vector = se3_lie_group.tangent_vector

        # Retrieve the inverse of the se3 group element
        inv_lie_se3_group = se3_lie_group.inverse()
        logging.info(f'\nSE3 element:\n{se3_lie_group}\nInverse:--\n{inv_lie_se3_group}')
        self.assertTrue(inv_lie_se3_group.this_group_element().shape == (4, 4))
        inv_tgt_vector = inv_lie_se3_group.tangent_vector

        vec_descriptor = ("       Original Algebra\n 0.000 0.000 1.000 1.000\n 0.000 0.000 0.000 0.000"
                          "\n-1.000 0.000 0.000 0.000\n 0.000 0.000 0.000 1.000")
        se3_element_desc = SE3ElementDescriptor(vec=tgt_vector,
                                                x=-2.4,
                                                y=-2.1,
                                                z=3.1,
                                                s=vec_descriptor,
                                                color='red')

        inv_vec_descriptor = ("       Inverse Algebra\n 0.540 0.000 0.841  1.510\n 0.000 1.000 0.000  0.000"
                              "\n-0.841 0.000 0.540 -0.668\n 0.000 0.000 0.000  2.718")
        se3_inv_element_desc = SE3ElementDescriptor(vec=inv_tgt_vector,
                                                    x=1.0,
                                                    y=0.1,
                                                    z=3.1,
                                                    s=inv_vec_descriptor,
                                                    color='blue')

        se3_visualization = SE3Visualization(rot_matrix=u3d.y_rot,trans_matrix=u3d.x_trans)
        se3_visualization.visualize(se3_element_descs=[se3_element_desc, se3_inv_element_desc],
                                    initial_point=se3_visualization.identity(),
                                    scale=(-1.0, 1.0),
                                    num_points=48,
                                    title='48 Displacement Points - SE(3) Inverse @ Identity')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_composition_visualization(self):
        first_trans = u3d.x_trans + u3d.y_trans + u3d.z_trans
        second_trans = u3d.x_trans - u3d.y_trans - 3
        se3_group_1 = SE3Visualization(rot_matrix=u3d.y_rot, trans_matrix=first_trans)
        se3_group_2 = SE3Visualization(rot_matrix=u3d.z_rot, trans_matrix=second_trans)

        # Composition
        se3_composed_group = se3_group_1.compose(se3_group_2)
        self.assertTrue(se3_composed_group.belongs())
        logging.info(f'\nFirst element:\n{se3_group_1}\nSecond element\n{se3_group_2}'
              f'\nComposed element: {se3_composed_group}')

        vec_descriptor_1 = ("          First Algebra\n 0.000 0.000 1.000 1.000\n 0.000 0.000 0.000 1.000"
                            "\n-1.000 0.000 0.000 1.000\n 0.000 0.000 0.000 1.000  ")
        se3_element_1_desc = SE3ElementDescriptor(vec=se3_group_1.tangent_vector,
                                                  x=-4.4,
                                                  y=-4.5,
                                                  z=-0.6,
                                                  s=vec_descriptor_1,
                                                  color='red')

        vec_descriptor_2 = ("         Second Algebra\n0.000 -1.000 0.000 -2.000\n1.000  0.000 0.000 -4.000"
                            "\n0.000  0.000 0.000 -3.000\n0.000  0.000 0.000  1.000 ")
        se3_element_2_desc = SE3ElementDescriptor(vec=se3_group_2.tangent_vector,
                                                  x=-4.3,
                                                  y=-4.5,
                                                  z=2.2,
                                                  s=vec_descriptor_2,
                                                  color='blue')

        composed_vec_descriptor = ("       Composed Algebra\n 0.292 -0.455 0.841  1.396\n 0.841  0.540 0.000 -2.705"
                                   "\n-0.455  0.708 0.540 -0.206\n 0.000  0.000 0.000  1.000")
        se3_composed_element_desc = SE3ElementDescriptor(vec=se3_composed_group.tangent_vector,
                                                         x=-4.2,
                                                         y=-4.5,
                                                         z=5.0,
                                                         s=composed_vec_descriptor,
                                                         color='green')

        se3_visualization = SE3Visualization(rot_matrix=u3d.y_rot, trans_matrix=u3d.x_trans)
        se3_visualization.visualize(se3_element_descs=[se3_element_1_desc,
                                                       se3_element_2_desc,
                                                       se3_composed_element_desc],
                                    initial_point=se3_visualization.identity(),
                                    num_points=64,
                                    scale=(-0.7, 0.7),
                                    title='64 Displacement Points - SE(3) Composition @ Identity')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_self_composition(self):
        se3_group = SE3Visualization(rot_matrix=u3d.x_rot, trans_matrix=u3d.y_trans)

        # Composition
        se3_composed_group = se3_group.compose(se3_group)
        self.assertTrue(se3_composed_group.belongs())

        vec_descriptor = ("              Algebra\n0.000 0.000  0.000 0.000\n0.000 0.000 -1.000 1.000"
                          "\n0.000 1.000  0.000 0.000\n0.000 0.000  0.000 1.000")
        se3_element_desc = SE3ElementDescriptor(vec=se3_group.tangent_vector,
                                                x=-3.1,
                                                y=-1.1,
                                                z=-0.4,
                                                s=vec_descriptor,
                                                color='red')

        composed_vec_descriptor = (" Self Composed Algebra\n1.000  0.000  0.000 0.000\n0.000 -0.416 -0.909 4.357"
                                   "\n0.000  0.989 -0.416 2.448\n0.000  0.000  0.000 1.000")
        se3_composed_element_desc = SE3ElementDescriptor(vec=se3_composed_group.tangent_vector,
                                                         x=-3.0,
                                                         y=-1.1,
                                                         z=1.3,
                                                         s=composed_vec_descriptor,
                                                         color='blue')
        se3_visualization = SE3Visualization(rot_matrix=u3d.y_rot, trans_matrix=u3d.x_trans)
        se3_visualization.visualize(se3_element_descs=[se3_element_desc, se3_composed_element_desc],
                                    num_points=48,
                                    initial_point=se3_visualization.identity(),
                                    scale=(-0.35, 0.35),
                                    title='48 Displacement Points - SE(3) Self Composition @ Identity')

    def test_animate(self):
        logging.info(f'\nRotation matrix:\n{np.reshape(u3d.y_rot, (3, 3))}')
        logging.info(f'Translation vector: {u3d.x_trans}')
        se3_visualization = SE3Visualization(u3d.y_rot, u3d.x_trans)

        rot_trans: List[float] = [1.9, 0.1, 0.35, 1.5, 1.5, 0.5]
        rot_trans_str = ', '.join([str(x) for x in rot_trans])
        se3_element_1 = SE3ElementDescriptor(vec=gs.array(rot_trans),
                                             x=0.0,
                                             y=-4.1,
                                             z=-4.0,
                                             s=f'Vector [{rot_trans_str}]',
                                             color='red')

        rot_trans: List[float] = [-1.6, 0.5, 2.5, -1.9, 2.2, 2.3]
        rot_trans_str = ', '.join([str(x) for x in rot_trans])
        se3_element_2 = SE3ElementDescriptor(vec=gs.array(rot_trans),
                                             x=-0.1,
                                             y=-4.0,
                                             z=-4.8,
                                             s=f'Vector [{rot_trans_str}]',
                                             color='blue')
        num_displacement_points = 128
        se3_visualization.animate(se3_element_descs=[se3_element_1, se3_element_2],
                                  num_points=num_displacement_points,
                                  initial_point=se3_visualization.identity(),
                                  scale=(-1.2, 1.2),
                                  title=f'{num_displacement_points} Displacements Points - SE(3) @ Identity',
                                  interval=2400,
                                  fps=8)

