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
# 3rd Party Library import
import numpy as np
# Library imports
from geometry.lie import LieException
from geometry.lie.lie_se3_group import LieSE3Group
from play import Play
import python

# Unit rotation and translation 3D matrices
x_rot = np.array([[0.0, 0.0, 0.0],  # Unit rotation along X axis
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])
y_rot = np.array([[0.0, 0.0, 1.0],  # Unit rotation along Y axis
                  [0.0, 0.0, 0.0],
                  [-1.0, 0.0, 0.0]])
z_rot = np.array([[0.0, -1.0, 0.0],  # Unit rotation along Z axis
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
x_trans = np.array([1.0, 0.0, 0.0])  # Unit translation along X axis
y_trans = np.array([0.0, 1.0, 0.0])  # Unit translation along Y axis
z_trans = np.array([0.0, 0.0, 1.0])  # Unit translation along Z axis


class LieSE3GroupPlay(Play):
    """
    Source code related to the Substack article 'SE(3): The Lie Group That Moves the World'
    Reference: https://patricknicolas.substack.com/p/se3-the-lie-group-that-moves-the

    The features are implemented in the class LieSE3Group, python/geometry/lie/lie_se3_group.py
    The class LieSE3GroupPlay is a wrapper of the class LieSE3Group
    The execution of the tests (main) follows the same order as in the Substack article
    """
    def __init__(self, rot_matrix: np.array, trans_matrix: np.array) -> None:
        self.lie_se3_group = LieSE3Group(rot_matrix=rot_matrix,
                                         trans_matrix=trans_matrix,
                                         point_type='matrix',
                                         epsilon=1e-4)
        super(LieSE3GroupPlay, self).__init__()

    def play(self) -> None:
        self.play_inverse()
        self.play_self_compose()

        other_translation = x_trans - y_trans - 3
        other_lie_se3_group = LieSE3Group(rot_matrix=z_rot, trans_matrix=other_translation)
        self.play_compose(other_lie_se3_group)

    def play_inverse(self) -> None:
        """
        Source code related to Substack article 'SE(3): The Lie Group That Moves the World' - Code snippet 6
        Ref: https://patricknicolas.substack.com/p/se3-the-lie-group-that-moves-the
        """
        tgt_vector = self.lie_se3_group.tangent_vector
        logging.info(tgt_vector)
        inv_lie_se3_group = self.lie_se3_group.inverse()

        logging.info(f'\nSE3 element:\n{self.lie_se3_group}\nInverse:--\n{inv_lie_se3_group}')
        assert inv_lie_se3_group.this_group_element().shape == (4, 4)
        inv_tgt_vector = inv_lie_se3_group.tangent_vector
        logging.info(inv_tgt_vector)

    def play_self_compose(self) -> None:
        """
        Source code related to Substack article 'SE(3): The Lie Group That Moves the World' - Code snippet 8
        Ref: https://patricknicolas.substack.com/p/se3-the-lie-group-that-moves-the
        """
        se3_composed_group = self.lie_se3_group.compose(self.lie_se3_group)
        logging.info(f'\nFirst element:\n{self.lie_se3_group}\nSelf Composed element: {se3_composed_group}')
        assert se3_composed_group.belongs() is True

    def play_compose(self, lie_se3_group_2: LieSE3Group) -> None:
        """
        Source code related to Substack article 'SE(3): The Lie Group That Moves the World' - Code snippet 9
        Ref: https://patricknicolas.substack.com/p/se3-the-lie-group-that-moves-the
        """
        # Composition
        se3_composed_group = self.lie_se3_group.compose(lie_se3_group_2)
        assert se3_composed_group.belongs()
        logging.info(f'\nFirst element:\n{self.lie_se3_group}\nSecond element\n{lie_se3_group_2}'
                     f'\nComposed element: {se3_composed_group}')


if __name__ == '__main__':
    try:
        # Test 1: 'SE(3): The Lie Group That Moves the World' - Code snippet 6
        lie_se3_group_play = LieSE3GroupPlay(rot_matrix=y_rot, trans_matrix=x_trans)
        lie_se3_group_play.play_inverse()

        # Test 2: 'SE(3): The Lie Group That Moves the World' - Code snippet 8
        lie_se3_group_play = LieSE3GroupPlay(rot_matrix=x_rot, trans_matrix=y_trans)
        lie_se3_group_play.play_self_compose()

        # Test 3:  'SE(3): The Lie Group That Moves the World' - Code snippet 9
        translation = x_trans + y_trans + z_trans
        lie_se3_group_play = LieSE3GroupPlay(rot_matrix=y_rot, trans_matrix=translation)
        translation_2 = x_trans - y_trans - 3
        lie_se3_group_play_2 = LieSE3GroupPlay(rot_matrix=z_rot, trans_matrix=translation_2)
        lie_se3_group_play.play_compose(lie_se3_group_play_2.lie_se3_group)
    except AssertionError as e:
        logging.error(e)
        assert False
    except ValueError as e:
        logging.error(e)
        assert False
    except LieException as e:
        logging.error(e)
        assert False