
import unittest
from Lie.Lie_SE3_group import VisualTangentVector
from Lie.SE3_visualization import SE3Visualization
import numpy as np
from typing import List
import geomstats.backend as gs


class SE3VisualizationTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_animation_one(self):
        from Lie import unit_elements

        visualization = SE3Visualization(unit_elements.y_rot, unit_elements.x_trans)
        print(visualization)

        rot_trans: List[float] = [-1.6, 0.5, 2.5, -1.9, 2.2, 2.3]
        rot_trans_str = ', '.join([str(x) for x in rot_trans])
        visual_tangent_vectors = VisualTangentVector(vec=gs.array(rot_trans),
                                                     x=-0.1,
                                                     y=-4.0,
                                                     z=-4.8,
                                                     s=f'Vector [{rot_trans_str}]',
                                                     color='red')
        num_points = 96
        visualization.animate(visual_tangent_vecs=[visual_tangent_vectors],
                              num_points=num_points,
                              title=f'{num_points} Displacements Points Animation - SE(3)',
                              interval=1000,
                              fps=40)

    @unittest.skip('Ignore')
    def test_animation_two(self):
        from Lie import unit_elements

        visualization = SE3Visualization(unit_elements.y_rot, unit_elements.x_trans)
        print(visualization)

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
        num_points = 128
        visualization.animate(visual_tangent_vecs=[visual_tangent_vectors_1, visual_tangent_vectors_2],
                              num_points=num_points,
                              title=f'{num_points} Displacements Points Animation - SE(3)',
                              interval=400,
                              fps=40)

    def test_build_from_unit_elements(self):
        from Lie import UnitElements

        epsilon = 1e-4
        print(f'\nRotation matrix:\n{np.reshape(UnitElements.y_rot, (3, 3))}')
        print(f'Translation vector: {UnitElements.x_trans}')
        lie_se3_group = SE3Visualization(UnitElements.y_rot, UnitElements.x_trans)
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


