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

import numpy as np
from typing import AnyStr, List, Self, Callable, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from util.base_animation import BaseAnimation
__all__ = ['SO3Animation', 'default_so3_transform']


def default_so3_transform(args: List[np.array]) -> np.array:
    theta = args[0]
    T = np.eye(4)
    T[:3, :3] = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return T


class SO3Animation(BaseAnimation):
    """
    Wrapper for simulation or animation of SO3 lie group transformation defined as
    math::
        \begin{matrix}
        cos(\theta) & -sin(\theta)  & 0 \\
        sin(\theta) &  cos(\theta) & 0  \\
        0 & 0 & 1  \\
        \end{matrix}

    The method uses FuncAnimation frame based simulator with the update (stepping) method implemented
    as a nested function.

    Reference: https://patricknicolas.substack.com/p/mastering-special-orthogonal-groups
    """
    def __init__(self,
                 logo_pos: List[float],
                 interval: int,
                 fps: int,
                 coordinates: (float, float, float),
                 transform: Callable[[np.array], np.array] = default_so3_transform,
                 sphere_radius: float = 1.0) -> None:
        """
        Default constructor for the animation of SO3 lie Group.

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
        @param sphere_radius: Radius of the 3D sphere
        @type sphere_radius: float
        """
        super(SO3Animation, self).__init__(logo_pos, interval, fps)

        self.coordinates = coordinates
        self.transform = transform
        self.next_step = [np.array(0.0), np.array([[0.0], [0.0], [0.0]])]
        self.fig = plt.figure(figsize=(10, 7))
        self.sphere_radius = sphere_radius
        self.ax = self.fig.add_subplot(111, projection='3d')

    @classmethod
    def build(cls,
              logo_pos: List[float],
              interval: int,
              fps: int,
              transform: Callable[[np.array], np.array] = default_so3_transform,
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
        x, y, z = SO3Animation._set_coordinates(sphere_radius)
        return cls(logo_pos=logo_pos,
                   interval=interval,
                   fps=fps, coordinates=(x, y, z),
                   transform=transform,
                   sphere_radius=sphere_radius)

    def draw(self, mp4_file: bool = False) -> None:
        """
        Draw and animate a 3D sphere in an ambient Euclidean space. The animation is driven by Matplotlib
        FuncAnimation class that require an update nested function.

        @param mp4_file: Flag to specify if the mp4 file is to be generated (False plot are displayed but not saved)
        @type mp4_file: boolean
        """
        colors = ['#cffa0d', '#8bfa0d', '#1ffa0d', '#0dfa9d', '#0dfae4', '#0ddafa', '#9cb1fe', '#bb9cfe', '#e59cfe',
                  '#fe9cd6', '#fc9e93', '#fe9cd6', '#e59cfe', '#bb9cfe', '#9cb1fe', '#0ddafa', '#0dfae4', '#0dfa9d',
                  '#1ffa0d', '#8bfa0d']
        colors = colors + colors + colors + colors
        trajectory_pts = np.linspace(0, np.pi, len(colors)+1)
        next_pts = [SO3Animation.__trajectory(t) for pt_index, t in enumerate(trajectory_pts)]

        points = np.stack([self.coordinates[0].ravel(),
                           self.coordinates[1].ravel(),
                           self.coordinates[2].ravel(),
                           np.ones(self.coordinates[0].size)])
        self.fig.patch.set_facecolor('#f0f9ff')
        self.ax.set_facecolor('#f0f9ff')
        self._draw_logo(fig=self.fig)
        self.__draw_formula()

        geo_lines = self.__sphere_geo_lines()

        def update(frame: int) -> None:
            """
            Update method to be executed for each frame
            @param frame: Number of the frame (index) used in the simulation
            @type frame: int
            """
            self.ax.clear()
            self.__reset_axis()
            self.__animation_step(next_pts[frame], frame)
            self._draw_trajectory(next_pts, frame)
            T = self.transform(self.next_step)
            self.__draw_next_sphere(colors[frame], geo_lines, points, T)

        self.__reset_axis()
        self._draw_trajectory(next_pts)
        self.__draw_sphere(color='#facf0d', geo_lines=geo_lines)

        ani = FuncAnimation(self.fig, update, frames=len(colors), interval=self.interval, repeat=False, blit=False)
        if mp4_file:
            file_name = f'{self._group_name()}_animation.mp4'
            ani.save(file_name, writer='ffmpeg', fps=self.fps, dpi=240)
        else:
            plt.show()

    """ ----------------  Polymorphic protected methods ------------------  """

    @staticmethod
    def _set_coordinates(sphere_radius: float) -> Tuple[np.array, np.array,  np.array]:
        u, v = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50)
        x = sphere_radius * np.outer(np.cos(u), np.sin(v))
        y = sphere_radius * np.outer(np.sin(u), np.sin(v))
        z = sphere_radius * np.outer(np.ones_like(u), np.cos(v))
        return x, y, z

    def _draw_trajectory(self, next_pts: np.array, frame: int = -1) -> None:
        pass

    def _group_name(self) -> AnyStr:
        return 'SO3'

    """ ------------------  Private Helper Methods -------------------  """

    def __animation_step(self, next_point: np.array, frame: int) -> None:
        self.next_step = [frame*0.1, next_point]

    def __draw_formula(self) -> None:
        import matplotlib.image as mpimg
        img = mpimg.imread(f'../../input/{self._group_name()}_formula.png')
        inset_ax = self.fig.add_axes((0.01, 0.35, 0.24, 0.24))
        inset_ax.imshow(img, alpha=1.0)
        inset_ax.axis('off')


    @staticmethod
    def __trajectory(t: np.array) -> np.array:
        import math
        radius = 1.2
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        u = (t/np.pi - 1/2)
        z = 12.5*math.exp(-u*u) - np.array(11)
        return np.array([[x], [y], [z]])

    def __draw_sphere(self, color: AnyStr, geo_lines: List[np.array]) -> None:
        self.ax.plot_surface(self.coordinates[0],
                             self.coordinates[1],
                             self.coordinates[2],
                             color=color,
                             linewidth=0,
                             alpha=0.6,
                             label='Original Sphere')
        for line in geo_lines:
            self.ax.plot(line[0], line[1], line[2], color='black', linewidth=1.5, alpha=1.0)

    def __draw_next_sphere(self, color: AnyStr, geo_lines: List[np.array], points: np.array, T) -> None:
        # Step 3: Apply SE(3) transformation
        transformed_points = T @ points
        self.coordinates = [transformed_points[idx].reshape(self.coordinates[idx].shape) for idx in range(3)]

        # Transformed sphere
        self.ax.plot_surface(self.coordinates[0],
                             self.coordinates[1],
                             self.coordinates[2],
                             color=color,
                             linewidth=0,
                             alpha=0.6,
                             label='New Sphere')
        for idx, line in enumerate(geo_lines):
            # Transformed line
            transformed_line = T @ line
            match idx:
                case 3:
                    color = 'red'
                    line_width = 4
                case 15:
                    color = 'black'
                    line_width = 4
                case _:
                    color = 'black'
                    line_width = 1
            self.ax.plot(transformed_line[0],
                         transformed_line[1],
                         transformed_line[2],
                         color=color,
                         linewidth=line_width,
                         alpha=1.0)

    def __sphere_geo_lines(self) -> List[np.array]:
        latitudes = np.linspace(-np.pi / 2, np.pi / 2, 9)
        longitudes = np.linspace(0, 2 * np.pi, 18)

        geo_lines = []
        for lat in latitudes:
            phi = np.linspace(0, 2 * np.pi, 100)
            x_lat = self.sphere_radius * np.cos(phi) * np.cos(lat)
            y_lat = self.sphere_radius * np.sin(phi) * np.cos(lat)
            z_lat = self.sphere_radius * np.ones_like(phi) * np.sin(lat)
            geo_lines.append(np.stack([x_lat, y_lat, z_lat, np.ones_like(phi)]))

        for lon in longitudes:
            theta = self.sphere_radius * np.linspace(-np.pi / 2, np.pi / 2, 100)
            x_lon = self.sphere_radius * np.cos(theta) * np.cos(lon)
            y_lon = self.sphere_radius * np.cos(theta) * np.sin(lon)
            z_lon = self.sphere_radius * np.sin(theta)
            geo_lines.append(np.stack([x_lon, y_lon, z_lon, np.ones_like(theta)]))
        return geo_lines

    def __reset_axis(self):
        self.ax.set_box_aspect([1.3, 1.3, 1.2])
        self.ax.set_xlim(-1.8, 1.8)
        self.ax.set_ylim(-1.8, 1.8)
        self.ax.set_zlim(-1.8, 1.8)
        self.ax.set_title(x=0.6,
                          y=1.0,
                          label='Mastering Special Orthogonal Groups-Python',
                          fontdict={'fontsize': 18, 'fontweight': 'bold', 'fontname': 'Helvetica', 'color': 'black'})
        self.ax.set_xlabel('X', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        self.ax.set_ylabel('Y', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        self.ax.set_zlabel('Z', fontdict={'fontsize': 14, 'fontweight': 'bold'})


if __name__ == '__main__':
    lie_group_simulation = SO3Animation.build(logo_pos=[0.015, 0.725, 0.3, 0.28],
                                              interval=2000,
                                              fps=10,
                                              sphere_radius=2)
    lie_group_simulation.draw(mp4_file=True)



