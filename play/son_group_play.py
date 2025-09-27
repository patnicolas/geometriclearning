__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Python standard library imports
import logging
import random
# 3rd Party Library import
import torch
# Library imports
from geometry.lie import LieException
from play import Play
from geometry.lie.son_group import SOnGroup
import python


class SOnGroupPlay(Play):
    """
    Source code related to the two Substack articles
    'A Journey into the Lie Group SO(4)'
        Reference: https://patricknicolas.substack.com/p/a-journey-into-the-lie-group-so4
    'Mastering Special Orthogonal Groups With Practice'
        Reference: https://patricknicolas.substack.com/p/mastering-special-orthogonal-groups

    The features are implemented in the class SOnGroup, python/geometry/lie/son_group.py
    The class SOnGroupPlay is a wrapper of the class SOnGroup
    The execution of the tests (main) follows the same order as in the Substack article
    """

    def __init__(self, dimension: int) -> None:
        assert 2 <= dimension <= 4, f'Dimension of the Lie group {dimension} should be [2, 4]'

        super(SOnGroupPlay, self).__init__()
        self.son_group = SOnGroup(dim=dimension, equip=True)

    def play(self) -> None:
        self.play_generate_rotation()

        match self.son_group.dimension():
            case 2:
                self.play_exp_log_map()
                self.play_inverse()
                self.play_projection()
            case 3:
                self.play_exp_log_map()
                self.play_composition()
            case 4:
                self.play_composition()
                self.play_inverse()
                self.play_projection()

    def play_generate_rotation(self) -> None:
        match self.son_group.dimension():
            case 2 | 3:
                """
                Substack article "Mastering Special Orthogonal Groups With Practice" Code snippet 4
                ref: https://patricknicolas.substack.com/p/mastering-special-orthogonal-groups
                """
                rotation = self.son_group.random_matrix()
                logging.info(f'\nSO({self.son_group.dimension()}) random matrix:\n{rotation}')
                SOnGroup.validate_points(rotation, dim=self.son_group.dimension())
            case 4:
                """
                Substack article "A Journey into the Lie Group SO(4)" Code snippet 3
                https://patricknicolas.substack.com/p/a-journey-into-the-lie-group-so4 
                """
                weights = [random.random() for _ in range(self.son_group.num_basis_matrices())]
                # Normalize weights distribution for basis matrices
                total_weights = sum(weights)
                weights = [w / total_weights for w in weights]
                # Generate so4 Algebra element
                so4_generated = self.son_group.generate_rotation(weights=weights)
                logging.info(f'\nGenerated SO4 element: {so4_generated}')
                assert so4_generated.shape == (4, 4)
            case _:
                raise ValueError(f'Dimension of Lie group: {self.son_group.dimension()} is out of range')

    def play_exp_log_map(self) -> None:
        match self.son_group.dimension():
            case 2:
                """
                Substack article "Mastering Special Orthogonal Groups With Practice" Code snippet 9
                ref: https://patricknicolas.substack.com/p/mastering-special-orthogonal-groups
                """
                base_point = torch.eye(2)
                # Given tangent vector
                tgt_vector = torch.Tensor([
                    [0, -0.707],
                    [0.707, 0]
                ])
            case 3:
                """
                Substack article "Mastering Special Orthogonal Groups With Practice" Code snippet 10
                ref: https://patricknicolas.substack.com/p/mastering-special-orthogonal-groups
                """
                base_point = torch.eye(3)
                # Given tangent vector
                tgt_vector = torch.Tensor([
                    [0, -1.0, 0.5],
                    [1.0, 0.0, -1.0],
                    [-0.5, 1.0, 0.0]
                ])
            case _:
                raise ValueError(f'Dimension of Lie group: {self.son_group.dimension()} is out of range')

        """
        Substack article "Mastering Special Orthogonal Groups With Practice" Code snippet 9 & 10
        ref: https://patricknicolas.substack.com/p/mastering-special-orthogonal-groups
        """
        # Apply the exponential map
        end_point = self.son_group.exp(tgt_vector, base_point)
        logging.info(f'SO({self.son_group.dimension()}) end point:\n{end_point}')

        # Validate the end point belongs to SO(2) or SO(3) groups
        SOnGroup.validate_points(end_point, dim=self.son_group.dimension())

        # Apply the logarithm map back to the algebra
        computed_tgt_vector = self.son_group.log(end_point, base_point)
        logging.info(f'so{self.son_group.dimension()} algebra A:\n{tgt_vector}'
                     f'\nso{self.son_group.dimension()} computed algebra log(exp(A)):\n{computed_tgt_vector}')

        # Verify the vector recomputed from the end point using the log map is similar to the original vector
        assert self.son_group.equal(tgt_vector, computed_tgt_vector)

    def play_composition(self) -> None:
        dimension = self.son_group.dimension()
        match dimension:
            case 3:
                """
                Substack article "Mastering Special Orthogonal Groups With Practice" Code snippet 11
                ref: https://patricknicolas.substack.com/p/mastering-special-orthogonal-groups
                """
                matrix1 = SOnGroupPlay.__create_matrix(rad=45, dim=3)
                matrix2 = SOnGroupPlay.__create_matrix(rad=0, dim=3)

                # Composed rotations
                composed_rotation = self.son_group.compose(matrix1, matrix2)
                logging.info(f'\nMatrix 1: {matrix1}\nMatrix 2: {matrix2}\nComposed matrix:\n{composed_rotation}')

                # Self composed rotations
                composed_rotation = self.son_group.compose(matrix1, matrix1)
                logging.info(f'\nMatrix: {matrix1}\nSelf composed matrix:\n{composed_rotation}')

            case 4:
                """
                Substack article "A Journey into the Lie Group SO(4)" Code snippet 4
                https://patricknicolas.substack.com/p/a-journey-into-the-lie-group-so4
                """
                # Generate rotations1
                rand_matrix1 = self.son_group.random_matrix()
                rand_matrix2 = self.son_group.random_matrix()
                # Validate rotations
                SOnGroup.validate_points(rand_matrix1, rand_matrix2, dim=dimension, rtol=1e-4)

                # Composed rotations
                composed_rotation = self.son_group.compose(rand_matrix1, rand_matrix2)
                SOnGroup.validate_points(composed_rotation, dim=dimension)
                logging.info(f'\nMatrix 1: {rand_matrix1}\nMatrix 2: {rand_matrix2}\nComposed matrix:\n{composed_rotation}')

                # Self composed rotations
                composed_rotation = self.son_group.compose(rand_matrix1, rand_matrix1)
                SOnGroup.validate_points(composed_rotation, dim=dimension)
                logging.info(f'\nMatrix 1: {rand_matrix1}\nSelf composed matrix:\n{composed_rotation}')
            case _:
                raise ValueError(f'Dimension of Lie group: {self.son_group.dimension()} should be 3 or 4')

    def play_inverse(self) -> None:
        dimension = self.son_group.dimension()
        match dimension:
            case 2:
                """
                Substack article "Mastering Special Orthogonal Groups With Practice" Code snippet 12
                ref: https://patricknicolas.substack.com/p/mastering-special-orthogonal-groups
                """
                matrix = SOnGroupPlay.__create_so_matrix(theta_rad=90, gamma_rad=None)
                # Validate rotation is SO(2)
                SOnGroup.validate_points(matrix, dim=dimension)

                inverse_matrix = self.son_group.inverse(matrix)
                # Validate inverse rotation is SO(2)
                SOnGroup.validate_points(inverse_matrix, dim=dimension)
                logging.info(f'\nMatrix:\n{matrix}\nInverse matrix:\n{inverse_matrix}')
                # Verify inverse
                identity = torch.eye(dimension)
                assert self.son_group.equal(matrix.T, inverse_matrix, 1e-5)
                inverse_identity = self.son_group.inverse(identity)
                # Validate inverse rotation is SO(2)
                SOnGroup.validate_points(inverse_identity, dim=dimension)
                logging.info(f'\nMatrix:\n{identity}\nInverse matrix:\n{inverse_identity}')

            case 4:
                """
                Substack article "A Journey into the Lie Group SO(4)" Code snippet 6
                https://patricknicolas.substack.com/p/a-journey-into-the-lie-group-so4
                """
                rotation = self.son_group.random_matrix()
                # Validate rotation is SO(4)
                SOnGroup.validate_points(rotation, dim=dimension)

                inverse_rotation = self.son_group.inverse(rotation)
                # Validate inverse rotation is SO(4)
                SOnGroup.validate_points(inverse_rotation, dim=dimension)
                logging.info(f'\nRotation:\n{rotation}\nInverse Rotation:\n{inverse_rotation}')

                # Verify inverse
                identity = torch.eye(dimension)
                self.son_group.equal(rotation.T @ inverse_rotation, identity)
            case _:
                raise ValueError(f'Dimension of Lie group: {self.son_group.dimension()} should be 2 or 4')

    def play_projection(self) -> None:
        from python import pretty_torch

        dimension = self.son_group.dimension()
        match dimension:
            case 3:
                """
                Substack article "Mastering Special Orthogonal Groups With Practice" Code snippet 13
                ref: https://patricknicolas.substack.com/p/mastering-special-orthogonal-groups
                """
                matrix = torch.Tensor(
                    [[0.0, 0.8, -1.0],
                     [-0.8, 0.0, 0.5],
                     [1.0, -0.5, 0.0]]
                )
                # Projected a given rotation matrix on SO(3) manifold
                projected_matrix = self.son_group.project(matrix)
                # Validate projected rotation matrix is SO(3)
                # SOnGroup.validate_points(projected_matrix, dim=3)
                logging.info(f'\nMatrix:\n{matrix}\nProjected matrix:\n{projected_matrix}')

                # Special case of projecting identity
                identity = torch.eye(3)
                projected_identity = self.son_group.project(identity)
                logging.info(f'\nIdentity:\n{identity}\nProjected identity:\n{projected_identity}')

            case 4:
                """
                Substack article "A Journey into the Lie Group SO(4)" Code snippet 8
                https://patricknicolas.substack.com/p/a-journey-into-the-lie-group-so4
                """
                matrix = torch.Tensor(
                    [[0.0, 0.8, -1.0, 0.3],
                     [-0.8, 0.0, 0.5, -0.4],
                     [1.0, -0.5, 0.0, 0.1],
                     [-0.3, 0.4, -0.1, 0.0]]
                )
                projected_matrix = self.son_group.project(matrix)
                logging.info('\nMatrix:')
                pretty_torch(matrix, w=8, d=4)
                logging.info('\nProjected matrix:')
                pretty_torch(projected_matrix)

                matrix = torch.eye(4)
                projected_matrix = self.son_group.project(matrix)
                SOnGroup.validate_points(projected_matrix, dim=4)
                logging.info(f'\nMatrix:\n{matrix}\nProjected matrix:\n{projected_matrix}')

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
    def __create_so_matrix(theta_rad: int, gamma_rad: int = None) -> torch.Tensor:
        import numpy as np
        theta = np.radians(theta_rad)
        if gamma_rad is None:
            return torch.Tensor([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
        else:
            gamma = np.radians(gamma_rad)
            return torch.Tensor([  # Rotation on X-axis
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ]) @ torch.Tensor([  # Rotation on Y-axis
                [np.cos(gamma), 0, np.sin(gamma)],
                [0, 1, 0],
                [-np.sin(gamma), 0, np.cos(gamma)]
                ])


if __name__ == '__main__':
    try:
        # Substack article "Mastering Special Orthogonal Groups With Practice"  SO(2)
        son_group_play = SOnGroupPlay(dimension=2)
        son_group_play.play_generate_rotation()
        son_group_play.play_exp_log_map()
        son_group_play.play_inverse()
        son_group_play.play_projection()

        # Substack article "Mastering Special Orthogonal Groups With Practice"  SO(3)
        son_group_play = SOnGroupPlay(dimension=3)
        son_group_play.play_generate_rotation()
        son_group_play.play_exp_log_map()
        son_group_play.play_composition()

        # Substack article 'A Journey into the Lie Group SO(4)'
        son_group_play = SOnGroupPlay(dimension=4)
        son_group_play.play_generate_rotation()
        son_group_play.play_composition()
        son_group_play.play_inverse()
        son_group_play.play_projection()
    except AssertionError as e:
        logging.error(e)
        assert False
    except ValueError as e:
        logging.error(e)
        assert False
    except LieException as e:
        logging.error(e)
        assert False




