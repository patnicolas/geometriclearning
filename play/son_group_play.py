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
    Source code related to the Substack article 'A Journey into the Lie Group SO(4)'
    The features are implemented in the class SOnGroup, python/geometry/lie/son_group.py
    The class SOnGroupPlay is a wrapper of the class SOnGrpu

    Reference: https://patricknicolas.substack.com/p/a-journey-into-the-lie-group-so4
    """

    def __init__(self, dimension: int) -> None:
        assert 2 < dimension < 6, f'Dimension of the Lie group {dimension} shoujld be [3, 5]'

        super(SOnGroupPlay, self).__init__()
        self.son_group = SOnGroup(dim=dimension, equip=True)

    def play(self) -> None:
        self.play_generate_rotation()
        self.play_composition()
        self.play_inverse()
        self.play_projection()

    def play_generate_rotation(self) -> None:
        """
        https://patricknicolas.substack.com/p/a-journey-into-the-lie-group-so4 Code snippet 3
        """
        weights = [random.random() for _ in range(self.son_group.num_basis_matrices())]
        # Normalize weights distribution for basis matrices
        total_weights = sum(weights)
        weights = [w / total_weights for w in weights]
        # Generate so4 Algebra element
        so4_generated = self.son_group.generate_rotation(weights=weights)
        logging.info(f'\nGenerated SO4 element: {so4_generated}')
        assert so4_generated.shape == (4, 4)

    def play_composition(self) -> None:
        """
        https://patricknicolas.substack.com/p/a-journey-into-the-lie-group-so4 code snippet 4
        """
        dimension = self.son_group.dimension()
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

    def play_inverse(self) -> None:
        """
        https://patricknicolas.substack.com/p/a-journey-into-the-lie-group-so4 code snippet 6
        """

        dim = self.son_group.dimension()
        rotation = self.son_group.random_matrix()
        # Validate rotation is SO(4)
        SOnGroup.validate_points(rotation, dim=dim)

        inverse_rotation = self.son_group.inverse(rotation)
        # Validate inverse rotation is SO(4)
        SOnGroup.validate_points(inverse_rotation, dim=dim)
        logging.info(f'\nRotation:\n{rotation}\nInverse Rotation:\n{inverse_rotation}')

        # Verify inverse
        identity = torch.eye(dim)
        self.son_group.equal(rotation.T @ inverse_rotation, identity)

    def play_projection(self) -> None:
        """
        https://patricknicolas.substack.com/p/a-journey-into-the-lie-group-so4 Code snippet 8
        """
        from python import pretty_torch

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


if __name__ == '__main__':
    try:
        son_group_play = SOnGroupPlay(dimension=4)
        son_group_play.play_generate_rotation()
        son_group_play.play_composition()
        son_group_play.play_inverse()
        son_group_play.play_projection()

    except AssertionError as e:
        logging.error(e)
        assert False
    except LieException as e:
        logging.error(e)
        assert False




