import unittest

from geometry.lie.son_group import SOnGroup
import logging
import torch
from geometry.lie import LieException
import os
from python import SKIP_REASON


class SOnGroupTest(unittest.TestCase):
    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_2(self):
        try:
            son_group = SOnGroup(dim=2, equip=False)
            logging.info(son_group)
            self.assertTrue(True)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except NotImplementedError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_3(self):
        try:
            son_group = SOnGroup(dim=3, equip=False)
            logging.info(son_group)
            sampled_points = son_group.sample_points(6)
            self.assertTrue(son_group.belongs(sampled_points))
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_4(self):
        try:
            son_group = SOnGroup(dim=4, equip=False)
            logging.info(son_group)
            logging.info(f'\n{son_group.sample_points(3)}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_exp_3(self):
        try:
            son_group = SOnGroup(dim=3, equip=False)
            logging.info(son_group)
            # Random uniform sampling of SO(3) rotation matrix
            sampled_point = son_group.sample_points(1)
            logging.info(f'Sampled point:\n{sampled_point}')
            # Given tangent vector
            tgt_vector = torch.Tensor(
                [[0.0, 0.8, -1.0],
                 [-0.8, 0.0, 0.5],
                 [1.0, -0.5, 0.0]]
            )
            self.assertTrue(son_group.belongs(sampled_point))
            end_point = son_group.exp(tgt_vector, sampled_point)
            logging.info(f'End point:\n{end_point}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_log_1(self):
        try:
            son_group = SOnGroup(dim=2, equip=False)
            # Identity base point
            base_point = torch.eye(2)
            # Given tangent vector
            tgt_vector = torch.Tensor([
                [0, -0.707],
                [0.707, 0]]
            )
            # Apply the exponential map
            end_point = son_group.exp(tgt_vector, base_point)
            logging.info(f'SO(2) end point:\n{end_point}')
            # Validate the end point belongs to SO(3) group
            SOnGroup.validate_points(end_point, dim=2)

            # Apply the logarithm map back to the algebra
            computed_tgt_vector = son_group.log(end_point, base_point)
            logging.info(f'so2 algebra A:\n{tgt_vector}'
                         f'\nso2 computed algebra log(exp(A)):\n{computed_tgt_vector}')
            # Verify the vector recomputed from the end point using the log map is similar to the original vector
            self.assertTrue(son_group.equal(tgt_vector, computed_tgt_vector))
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_log_3(self):
        try:
            son_group = SOnGroup(dim=3, equip=False)
            # Random uniform sampling of SO(3) rotation matrix
            identity = torch.eye(3)
            # Given tangent vector
            tgt_vector = torch.Tensor([
                [0, -1, 0.5],
                [1, 0, -1],
                [-0.5, 1, 0]]
            )
            # Apply the exponential map
            end_point = son_group.exp(tgt_vector, identity)
            logging.info(f'SO(3) end point:\n{end_point}')
            SOnGroup.validate_points(end_point, dim=3)

            # Apply the logarithm map back to the algebra
            computed_tgt_vector = son_group.log(end_point, identity)
            logging.info(f'so3 algebra A:\n{tgt_vector}\nso3 computed algebra log(exp(A)):\n{computed_tgt_vector}')
            self.assertTrue(son_group.equal(tgt_vector, computed_tgt_vector))
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
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
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
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
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
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
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_lie_algebra_4(self):
        try:
            son_group = SOnGroup(dim=4, equip=False)
            logging.info(son_group)
            sampled_point = son_group.sample_points(1)
            self.assertTrue(son_group.belongs(sampled_point))
            algebra = son_group.lie_algebra(sampled_point)
            logging.info(f'Algebra SO(4): {algebra}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_projection_1(self):
        try:
            son_group = SOnGroup(dim=2, equip=True)
            matrix = torch.Tensor(
                [[0.0, 0.8],
                 [-0.8, 0.0]]
            )
            projected_matrix = son_group.project(matrix)
            logging.info(f'\nMatrix\n{matrix}\nProjected matrix:\n{projected_matrix}')

            matrix = torch.Tensor(
                [[0.2, 0.7],
                 [-0.8, 0.2]]
            )
            projected_matrix = son_group.project(matrix)
            # Validate projected rotation matrix is SO(2)
            SOnGroup.validate_points(projected_matrix, dim=2)

            logging.info(f'\nMatrix\n{matrix}\nProjected matrix:\n{projected_matrix}')
            matrix = torch.eye(2)
            projected_matrix = son_group.project(matrix)
            logging.info(f'\nMatrix\n{matrix}\nProjected matrix:\n{projected_matrix}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_projection_2(self):
        try:
            son_group = SOnGroup(dim=3, equip=True)
            matrix = torch.Tensor(
                [[0.0, 0.8, -1.0],
                 [-0.8, 0.0, 0.5],
                 [1.0, -0.5, 0.0]]
            )
            # Projected a given rotation matrix on SO(3) manifold
            projected_matrix = son_group.project(matrix)
            # Validate projected rotation matrix is SO(3)
            # SOnGroup.validate_points(projected_matrix, dim=3)
            logging.info(f'\nMatrix:\n{matrix}\nProjected matrix:\n{projected_matrix}')

            # Special case of projecting identity
            identity = torch.eye(3)
            projected_identity = son_group.project(identity)
            logging.info(f'\nIdentity:\n{identity}\nProjected identity:\n{projected_identity}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_projection_3(self):
        try:
            from python import pretty_torch

            son_group = SOnGroup(dim=4, equip=True)
            matrix = torch.Tensor(
                [[0.0, 0.8, -1.0, 0.3],
                 [-0.8, 0.0, 0.5, -0.4],
                 [1.0, -0.5, 0.0, 0.1],
                 [-0.3, 0.4, -0.1, 0.0]]
            )
            projected_matrix = son_group.project(matrix)
            logging.info('\nMatrix:')
            pretty_torch(matrix, w=8, d=4)
            logging.info('\nProjected matrix:')
            pretty_torch(projected_matrix)

            matrix = torch.eye(4)
            projected_matrix = son_group.project(matrix)
            SOnGroup.validate_points(projected_matrix, dim=4)
            logging.info(f'\nMatrix:\n{matrix}\nProjected matrix:\n{projected_matrix}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_compose_1(self):
        try:
            son_group = SOnGroup(dim=3, equip=True)
            # Generate rotations1
            matrix1 = SOnGroupTest.__create_matrix(rad=45, dim=3)
            matrix2 = SOnGroupTest.__create_matrix(rad=0, dim=3)

            # Composed rotations
            composed_rotation = son_group.compose(matrix1, matrix2)
            logging.info(f'\nMatrix 1: {matrix1}\nMatrix 2: {matrix2}\nComposed matrix:\n{composed_rotation}')

            # Self composed rotations
            composed_rotation = son_group.compose(matrix1, matrix1)
            logging.info(f'\nMatrix: {matrix1}\nSelf composed matrix:\n{composed_rotation}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_compose_commutative(self):
        try:
            dimension = 4
            son_group = SOnGroup(dim=dimension, equip=True)
            # Generate rotations matrices
            rand_matrix1 = son_group.random_matrix()
            rand_matrix2 = son_group.random_matrix()

            composed_rotation1 = son_group.compose(rand_matrix1, rand_matrix2)
            composed_rotation2 = son_group.compose(rand_matrix2, rand_matrix1)
            logging.info(f'\nMatrix 1:\n{rand_matrix1}\nMatrix 2:\n{rand_matrix2}'
                         f'\nComposition Matrix 1 x matrix 2\n{composed_rotation1}'
                         f'\nComposition Matrix 1 x matrix 2\n{composed_rotation2}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_compose_2(self):
        try:
            dimension = 4
            son_group = SOnGroup(dim=dimension, equip=True)
            # Generate rotations1
            rand_matrix1 = son_group.random_matrix()
            rand_matrix2 = son_group.random_matrix()
            # Validate rotations
            SOnGroup.validate_points(rand_matrix1, rand_matrix2, dim=dimension, rtol=1e-4)

            # Composed rotations
            composed_rotation = son_group.compose(rand_matrix1, rand_matrix2)
            SOnGroup.validate_points(composed_rotation, dim=dimension)
            logging.info(f'\nMatrix 1: {rand_matrix1}\nMatrix 2: {rand_matrix2}\nComposed matrix:\n{composed_rotation}')

            # Self composed rotations
            composed_rotation = son_group.compose(rand_matrix1, rand_matrix1)
            SOnGroup.validate_points(composed_rotation, dim=dimension)
            logging.info(f'\nMatrix 1: {rand_matrix1}\nSelf composed matrix:\n{composed_rotation}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_inverse_1(self):
        try:
            group_dim = 2
            so2_group = SOnGroup(dim=group_dim, equip=True, atol=1e-4)
            matrix = SOnGroupTest.create_so_matrix(theta_rad=90, gamma_rad=None)
            # Validate rotation is SO(2)
            SOnGroup.validate_points(matrix, dim=group_dim)

            inverse_matrix = so2_group.inverse(matrix)
            # Validate inverse rotation is SO(2)
            SOnGroup.validate_points(inverse_matrix, dim=group_dim)
            logging.info(f'\nMatrix:\n{matrix}\nInverse matrix:\n{inverse_matrix}')
            # Verify inverse
            identity = torch.eye(group_dim)
            self.assertTrue(so2_group.equal(matrix.T,  inverse_matrix, 1e-5))
            inverse_identity = so2_group.inverse(identity)
            # Validate inverse rotation is SO(2)
            SOnGroup.validate_points(inverse_identity, dim=group_dim)
            logging.info(f'\nMatrix:\n{identity}\nInverse matrix:\n{inverse_identity}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_inverse_2(self):
        try:
            dim = 4
            son_group = SOnGroup(dim=dim, equip=True)
            rotation = son_group.random_matrix()
            # Validate rotation is SO(4)
            SOnGroup.validate_points(rotation, dim=dim)

            inverse_rotation = son_group.inverse(rotation)
            # Validate inverse rotation is SO(4)
            SOnGroup.validate_points(inverse_rotation, dim=dim)
            logging.info(f'\nRotation:\n{rotation}\nInverse Rotation:\n{inverse_rotation}')

            # Verify inverse
            identity = torch.eye(dim)
            son_group.equal(rotation.T @ inverse_rotation, identity)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_inverse_3(self):
        try:
            son_group = SOnGroup(dim=3, equip=True)
            matrix = torch.Tensor([[0.000, -1.000, 0.000],
                                   [0.000,  0.000, 1.000],
                                   [-1.000, 0.000, 0.000]])
            inverse_matrix = son_group.inverse(matrix)
            logging.info(f'\nMatrix:\n{matrix}\nInverse matrix:\n{inverse_matrix}')
            # Composition with its inverse
            composed = son_group.compose(matrix, inverse_matrix)
            logging.info(f'\nA={matrix}\nA o inv(A)={composed}')
            self.assertTrue(son_group.equal(composed, torch.eye(3)))
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_generate_rotation_1(self):
        try:
            son_group = SOnGroup(dim=3, equip=True)
            so3_generated = son_group.generate_rotation(weights=[0.2, 0.5, 0.3])
            logging.info(f'\nGenerated rotation: {so3_generated}')
            self.assertTrue(so3_generated.shape == (3, 3))
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_generate_rotation_2(self):
        try:
            son_group = SOnGroup(dim=4, equip=True)
            so4_generated = son_group.generate_rotation(weights=[0.2, 0.2, 0.2, 0.1, 0.1, 0.2])
            logging.info(f'\nGenerated SO4 element: {so4_generated}')
            self.assertTrue(so4_generated.shape == (4, 4))
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_generate_rotation_3(self):
        import random

        try:
            son_group = SOnGroup(dim=4, equip=True)
            weights = [random.random() for _ in range(son_group.num_basis_matrices())]
            # Normalize weights distribution for basis matrices
            total_weights = sum(weights)
            weights = [w/total_weights for w in weights]
            # Generate so4 Algebra element
            so4_generated = son_group.generate_rotation(weights=weights)
            logging.info(f'\nGenerated SO4 element: {so4_generated}')
            self.assertTrue(so4_generated.shape == (4, 4))
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except LieException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_create_matrix(self):
        matrix = SOnGroupTest.create_so_matrix(theta_rad=90)
        logging.info(f'\nSO(2) matrix read=45:\n{matrix}')
        matrix = SOnGroupTest.create_so_matrix(theta_rad=90, gamma_rad=45)
        logging.info(f'\nSO(3) matrix read=90:\n{matrix}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_create_random_matrix(self):
        # Random generation SO(2)
        son_group = SOnGroup(dim=2, equip=True)
        matrix = son_group.random_matrix()
        logging.info(f'\nSO(2) random matrix\n{matrix}')
        SOnGroup.validate_points(matrix, dim=2)

        # Random generation SO(3)
        son_group = SOnGroup(dim=3, equip=True)
        matrix = son_group.random_matrix()
        logging.info(f'\nSO(3) random matrix\n{matrix}')
        SOnGroup.validate_points(matrix, dim=3)

        # Random generation SO(4)
        son_group = SOnGroup(dim=4, equip=True)
        matrix = son_group.random_matrix()
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
    def create_so_matrix(theta_rad: int, gamma_rad: int = None) -> torch.Tensor:
        """
        Generate a SO(2) if gamma_rad is undefined/None or SO(3) matrix given two
        rotation angle theta and gamma

        Generate a SO2-SO3 matrix with theta and Gamma angles
        @param theta_rad: First angle in radians
        @type theta_rad: int
        @param gamma_rad: Second angle variable in radians
        type gamma_rad: int
        return: SO(2) or SO(3) matrix
        rtype: Torch Tensor
        """
        import numpy as np
        theta = np.radians(theta_rad)
        if gamma_rad is None:
            return torch.Tensor([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]
                ])
        else:
            gamma = np.radians(gamma_rad)
            return torch.Tensor([    # Rotation on X-axis
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ]) @ torch.Tensor([  # Rotation on Y-axis
                    [np.cos(gamma), 0, np.sin(gamma)],
                    [0, 1, 0],
                    [-np.sin(gamma), 0, np.cos(gamma)]
                ])

