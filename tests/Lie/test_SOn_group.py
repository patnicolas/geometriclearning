import unittest

from Lie.SOn_group import SOnGroup
import logging
import torch
import util
from geometry import GeometricException


class SOnGroupTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init_2(self):
        son_group = SOnGroup(dim=2, equip=False)
        logging.info(son_group)
        logging.info(son_group.sample_points(2))

    @unittest.skip('Ignore')
    def test_init_3(self):
        son_group = SOnGroup(dim=3, equip=False)
        logging.info(son_group)
        sampled_points = son_group.sample_points(6)
        self.assertTrue(son_group.belongs(sampled_points))

    @unittest.skip('Ignore')
    def test_init_4(self):
        son_group = SOnGroup(dim=4, equip=False)
        logging.info(son_group)
        logging.info(f'\n{son_group.sample_points(3)}')

    @unittest.skip('Ignore')
    def test_exp_3(self):
        try:
            son_group = SOnGroup(dim=3, equip=False)
            logging.info(son_group)
            sampled_point = son_group.sample_points(1)
            logging.info(f'Sampled point:\n{sampled_point}')
            tgt_vector = torch.Tensor(
                [[0.0, 0.8, -1.0],
                 [-0.8, 0.0, 0.5],
                 [1.0, -0.5, 0.0]]
            )
            self.assertTrue(son_group.belongs(sampled_point))
            end_point = son_group.exp(tgt_vector, sampled_point)
            logging.info(f'End point:\n{end_point}')
        except GeometricException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_log_3(self):
        try:
            son_group = SOnGroup(dim=3, equip=False)
            base_point = son_group.sample_points(1)
            tgt_vector = torch.Tensor(
                [[0.0, 0.8, -1.0],
                 [-0.8, 0.0, 0.5],
                 [1.0, -0.5, 0.0]]
            )
            end_point = son_group.exp(tgt_vector, base_point)
            logging.info(f'End point:\n{end_point}')
            computed_tgt_vector = son_group.log(end_point, base_point)
            logging.info(f'Original vector:\n{tgt_vector}\nComputed tangent vector:\n{computed_tgt_vector}')
            self.assertTrue(son_group.equal(tgt_vector, computed_tgt_vector))
        except GeometricException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_exp_4(self):
        try:
            son_group = SOnGroup(dim=4, equip=False)
            logging.info(son_group)
            sampled_point = son_group.sample_points(1)
            logging.info(f'Sampled point:\n{sampled_point}')
            tgt_vector = torch.Tensor(
                [[0.0, 0.8, -1.0, 0.3],
                 [-0.8, 0.0, 0.5, -0.4],
                 [1.0, -0.5, 0.0, 0.1],
                 [-0.3, 0.4, -0.1, 0.0]]
            )
            self.assertTrue(son_group.belongs(sampled_point))
            end_point = son_group.exp(tgt_vector, sampled_point)
            logging.info(f'End point:\n{end_point}')
        except GeometricException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_lie_algebra_3(self):
        try:
            son_group = SOnGroup(dim=3, equip=True)
            logging.info(son_group)
            sampled_point = son_group.sample_points(1)
            logging.info(f'Sampled point:\n{sampled_point}')
            self.assertTrue(son_group.belongs(sampled_point))
            tgt_vector = torch.Tensor(
                [[0.0, 0.8, -1.0],
                 [-0.8, 0.0, 0.5],
                 [1.0, -0.5, 0.0]]
            )
            end_point = son_group.exp(tgt_vector, sampled_point)
            rotation_matrix = son_group.lie_algebra(end_point)
            logging.info(f'Algebra SO(3): {rotation_matrix}')
        except GeometricException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_lie_algebra_2(self):
        try:
            son_group = SOnGroup(dim=2, equip=False)
            tgt_vector = torch.Tensor(
                [[0.0, 0.8],
                 [-0.8, 0.0]]
            )
            logging.info(son_group)
            sampled_point = son_group.sample_points(1)
            logging.info(f'Sampled point:\n{sampled_point}')
            self.assertTrue(son_group.belongs(sampled_point))
            end_point = son_group.exp(tgt_vector, sampled_point)
            rotation_matrix = son_group.lie_algebra(end_point)
            logging.info(f'Algebra SO(2): {rotation_matrix}')
        except GeometricException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_lie_algebra_4(self):
        try:
            son_group = SOnGroup(dim=4, equip=False)
            logging.info(son_group)
            sampled_point = son_group.sample_points(1)
            self.assertTrue(son_group.belongs(sampled_point))
            algebra = son_group.lie_algebra(sampled_point)
            logging.info(f'Algebra SO(4): {algebra}')
        except GeometricException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_projection_1(self):
        try:
            son_group = SOnGroup(dim=2, equip=True)
            matrix = torch.Tensor(
                [[0.0, 0.8],
                 [-0.8, 0.0]]
            )
            projected_matrix = son_group.projection(matrix)
            logging.info(f'\nMatrix\n{matrix}\nProjected matrix:\n{projected_matrix}')

            matrix = torch.Tensor(
                [[0.2, 0.7],
                 [-0.8, 0.2]]
            )
            projected_matrix = son_group.projection(matrix)
            # Validate projected rotation matrix is SO(2)
            SOnGroup.validate_son_input(projected_matrix, dim=2)

            logging.info(f'\nMatrix\n{matrix}\nProjected matrix:\n{projected_matrix}')
            matrix = torch.eye(2)
            projected_matrix = son_group.projection(matrix)
            logging.info(f'\nMatrix\n{matrix}\nProjected matrix:\n{projected_matrix}')
        except GeometricException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_projection_2(self):
        try:
            son_group = SOnGroup(dim=3, equip=True)
            matrix = torch.Tensor(
                [[0.0, 0.8, -1.0],
                 [-0.8, 0.0, 0.5],
                 [1.0, -0.5, 0.0]]
            )
            projected_matrix = son_group.projection(matrix)
            # Validate projected rotation matrix is SO(3)
            SOnGroup.validate_son_input(projected_matrix, dim=3)
            logging.info(f'\nMatrix:\n{matrix}\nProjected matrix:\n{projected_matrix}')

            matrix = torch.eye(3)
            projected_matrix = son_group.projection(matrix)
            logging.info(f'\nMatrix:\n{matrix}\nProjected matrix:\n{projected_matrix}')
        except GeometricException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_projection_3(self):
        try:
            son_group = SOnGroup(dim=4, equip=True)
            matrix = torch.Tensor(
                [[0.0, 0.8, -1.0, 0.3],
                 [-0.8, 0.0, 0.5, -0.4],
                 [1.0, -0.5, 0.0, 0.1],
                 [-0.3, 0.4, -0.1, 0.0]]
            )
            projected_matrix = son_group.projection(matrix)
            # Validate projected rotation matrix is SO(4)
            SOnGroup.validate_son_input(projected_matrix, dim=4)
            logging.info(f'\nMatrix:\n{matrix}\nProjected matrix:\n{projected_matrix}')

            matrix = torch.eye(4)
            projected_matrix = son_group.projection(matrix)
            SOnGroup.validate_son_input(projected_matrix, dim=4)
            logging.info(f'\nMatrix:\n{matrix}\nProjected matrix:\n{projected_matrix}')
        except GeometricException as e:
            logging.error(e)
            self.assertTrue(False)

    # @unittest.skip('Ignore')
    def test_inverse_1(self):
        try:
            son_group = SOnGroup(dim=3, equip=True)
            matrix = SOnGroupTest.__create_matrix(rad=90, dim=3)
            # Validate rotation is SO(3)
            SOnGroup.validate_son_input(matrix, dim=3)

            inverse_matrix = son_group.inverse(matrix)
            # Validate inverse rotation is SO(3)
            SOnGroup.validate_son_input(inverse_matrix, dim=3)
            logging.info(f'\nMatrix:\n{matrix}\nInverse matrix:\n{inverse_matrix}')
            # Verify inverse

            matrix = torch.eye(3)
            inverse_matrix = son_group.inverse(matrix)
            # Validate inverse rotation is SO(3)
            SOnGroup.validate_son_input(inverse_matrix, dim=3)
            logging.info(f'\nMatrix:\n{matrix}\nInverse matrix:\n{inverse_matrix}')
        except GeometricException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_inverse_2(self):
        try:
            dim = 4
            son_group = SOnGroup(dim=dim, equip=True)
            matrix = SOnGroupTest.__create_random_matrix(dim=dim)
            # Validate rotation is SO(4)
            SOnGroup.validate_son_input(matrix, dim=dim)

            inverse_matrix = son_group.inverse(matrix)
            # Validate inverse rotation is SO(4)
            SOnGroup.validate_son_input(inverse_matrix, dim=dim)
            logging.info(f'\nMatrix:\n{matrix}\nInverse matrix:\n{inverse_matrix}')

            matrix = torch.eye(4)
            inverse_matrix = son_group.inverse(matrix)
            # Validate inverse rotation is SO(4)
            SOnGroup.validate_son_input(inverse_matrix, dim=dim)
            logging.info(f'\nMatrix:\n{matrix}\nInverse matrix:\n{inverse_matrix}')
        except GeometricException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_create_matrix(self):
        matrix = SOnGroupTest.__create_matrix(45, 2)
        logging.info(f'\nSO(2) matrix read=45:\n{matrix}')
        matrix = SOnGroupTest.__create_matrix(90, 3)
        logging.info(f'\nSO(3) matrix read=90:\n{matrix}')

    @unittest.skip('Ignore')
    def test_create_random_matrix(self):
        matrix = SOnGroupTest.__create_random_matrix(2)
        logging.info(f'\nSO(2) random matrix\n{matrix}')
        SOnGroup.validate_son_input(matrix, dim=2)
        matrix = SOnGroupTest.__create_random_matrix(3)
        logging.info(f'\nSO(3) random matrix\n{matrix}')
        matrix = SOnGroupTest.__create_random_matrix(4)
        logging.info(f'\nSO(4) random matrix\n{matrix}')

    @staticmethod
    def __create_matrix(rad: int, dim: int) -> torch.Tensor:
        import numpy as np
        assert dim in (2, 3)
        theta = np.radians(rad)
        return torch.Tensor([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]
                ]) if dim == 2 else torch.Tensor([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])

    @staticmethod
    def __create_random_matrix(dim: int) -> torch.Tensor:
        A = torch.rand(dim, dim)
        Q, R = torch.linalg.qr(A)
        if torch.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        return Q
