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

from geometry.lie.animation.so3_animation import SO3Animation
import numpy as np
from typing import List, Callable, AnyStr, Dict, Any
__all__ = ['SE3Animation']


def default_se3_transform(args: List[np.array]) -> np.array:
    theta = args[0]
    t = args[1]
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    # Assemble the 4x4 SE(3) matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T

class SE3Animation(SO3Animation):
    """
    Wrapper for simulation or animation of SE3 lie group transformation defined as
    math::
        \begin{matrix}
        cos(\theta) & -sin(\theta)  & 0 & t_{1} \\
        sin(\theta) &  cos(\theta) & 0 & t_{2} \\
        0 & 0 & 1 & t_{3} \\
        0 & 0 & 0 & 1 \\
        \end{matrix}

    The method uses FuncAnimation frame based simulator with the update (stepping) method implemented
    as a nested function.

    Reference:

    Dictionary of animation configuration parameters
    ------------------------------------------------
    logo_pos: Tuple[int, int]   Position of the logo if one is defined
    logo_size: Tuple[int, int]  Size of the logo if one is defined
    interval: int  Interval for FuncAnimation in msec
    fps: int  Frame per second
    sphere_radius: float  Radius of sphere in 3D space
    x_lim: Tuple[float, float]  Range of x values
    y_lim: Tuple[float, float]  Range of y values
    z_lim: Tuple[float, float]  Range of z values
    formula_pos: Tuple[float, float]  Position of formula if any
    title_pos: Tuple[float, float]  Position of title
    """
    def __init__(self,
                 transform: Callable[[np.array], np.array] = default_se3_transform,
                 **kwargs: Dict[AnyStr, Any]) -> None:
        """
        Default constructor for the animation of SE3 lie Group.
        @param transform: Rotation (SO3) + Translation transform
        @type transform: Callable
        @param **kwargs: Dictionary of configuration parameters for any given animation
        @type **kwargs: Dictionary
        """

        super(SE3Animation, self).__init__(transform, **kwargs)

    """ ----------------  Polymorphic protected methods ------------------  """

    def _group_name(self) -> AnyStr:
        return 'SE3'

    def _draw_trajectory(self, next_pts: np.array, frame: int = -1) -> None:
        self.ax.plot([next_pts[0][0], np.array([0.0])],
                     [next_pts[0][1], np.array([0.0])],
                     [next_pts[0][2], np.array([0.0])],
                     color='blue',
                     linewidth=2,
                     label='Line between points')
        if frame > -1:
            for idx in range(frame + 1):
                next_pt = next_pts[idx + 1]
                pt = next_pts[idx]
                self.ax.plot([next_pt[0], pt[0]],
                             [next_pt[1], pt[1]],
                             [next_pt[2], pt[2]],
                             color='blue',
                             linewidth=2,
                             label='Line between points')


if __name__ == '__main__':
    config = {
        'fig_size': (10, 8),
        'logo_pos': (0.1, 0.97),
        'formula_pos': (0.01, 0.35),
        'formula_size': (0.24, 0.24),
        'title_pos': (0.6, 1.0),
        'x_lim': (-1.8, 1.8),
        'y_lim': (-1.8, 1.8),
        'z_lim': (-1.8, 1.8),
        'sphere_radius': 1.2,
        'interval': 1000,
        'fps': 8
    }
    lie_group_simulation = SE3Animation(transform=default_se3_transform, kwargs=config)
    lie_group_simulation.draw(mp4_file=True)

