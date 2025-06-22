__author__ = "Patrick Nicolas"
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

from lie.animation.so3_animation import SO3Animation
import numpy as np
from typing import List, Self, Callable, AnyStr
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
    """
    def __init__(self,
                 logo_pos: List[float],
                 interval: int,
                 fps: int,
                 coordinates: (float, float, float),
                 transform: Callable[[np.array], np.array] = default_se3_transform) -> None:
        """
              Default constructor for the animation of SE3 lie Group.

              @param logo_pos: Define the position of the chart [x, y, width, height]
              @type logo_pos: List[float]
              @param interval: Interval in milliseconds between frames
              @type interval: int
              @param fps: Number of frame per seconds for animation
              @type fps: int
              @param coordinates: Initial coordinate of the sphere used for SE3 transformation
              @type coordinates: Tuple[float, float,float]
              @param transform: Rotation (SO3) + Translation transform
              @type transform: Callable
              """
        super(SE3Animation, self).__init__(logo_pos, interval, fps, coordinates, transform, 1.0)

    @classmethod
    def build(cls,
              logo_pos: List[float],
              interval: int,
              fps: int,
              transform: Callable[[np.array], np.array] = default_se3_transform,
              sphere_radius: float = 1.0) -> Self:
        """
        Alternative constructor that takes a SE3 transformation as argument
        @param logo_pos: Define the position of the chart [x, y, width, height]
        @type logo_pos: List[float]
        @param interval: Interval in milliseconds between frames
        @type interval: int
        @param fps: Number of frame per seconds for animation
        @type fps: int
        @param transform: Rotation (SO3) + Translation transform
        @type transform: Callable
        @param sphere_radius: Radius of the 3D sphere
        @type sphere_radius: float
        @return: Instance of SE3Animation
        @rtype: SO3Animation
        """
        x, y, z = SO3Animation._set_coordinates(1.0)
        return cls(logo_pos=logo_pos,
                   interval=interval,
                   fps=fps, coordinates=(x, y, z),
                   transform=transform)

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
    lie_group_simulation = SE3Animation.build(logo_pos=[0.01, 0.74, 0.34, 0.24],
                                              interval=1000,
                                              fps=8,
                                              sphere_radius=1.6)
    lie_group_simulation.draw(mp4_file=True)

