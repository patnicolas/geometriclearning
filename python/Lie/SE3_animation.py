__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import numpy as np
from typing import AnyStr, List, Self, Callable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging
logger = logging.getLogger('Lie.SE3Animation')
__all__ = ['SE3Animation']


class SE3Animation(object):
    """
    Wrapper for simulation or animation of SE3 Lie group transformation defined as
    ..math::
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
                 coordinates: (float, float, float),
                 SE3_transform: Callable[[np.array], np.array],
                 interval: int,
                 fps: int) -> None:
        """
        Default constructor for the SE3 lie Group
        @param coordinates: Initial coordinate of the sphere used for SE3 transformation
        @type coordinates: Tuple[float, float,float]
        @param SE3_transform: Rotation (SO3) + Translation transform
        @type SE3_transform: Callable
        @param interval: Interval in milliseconds between frames
        @type interval: int
        @param fps: Number of frame per seconds for animation
        @type fps: int
        """
        self.coordinates = coordinates
        self.transform = SE3_transform
        self.interval = interval
        self.fps = fps
        self.next_step = [np.array(0.0), np.array([[0.0], [0.0], [0.0]])]
        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.patch.set_facecolor('lightblue')
        self.ax.set_facecolor('lightblue')
        self.ax.set_position([-0.55, 0.0, 2.0, 0.92])

    @classmethod
    def build(cls, transform: Callable[[np.array], np.array], interval: int, fps: int) -> Self:
        """
        Alternative constructor that takes a SE3 transformation as argument
        @param transform: Rotation (SO3) + Translation transform
        @type transform: Callable
        @return: Instance of SE3Animation
        @rtype: SE3Animation
        @param interval: Interval in milliseconds between frames
        @type interval: int
        @param fps: Number of frame per seconds for animation
        @type fps: int
        """
        u, v = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        return cls((x, y, z), transform, interval, fps)

    def draw(self) -> None:
        """
        Draw and animate a 3D sphere in an ambient Euclidean space. The animation is driven by Mathplotlib
        FuncAnimation class that require an update nested function.
        """
        colors = ['#cffa0d', '#8bfa0d', '#1ffa0d', '#0dfa9d', '#0dfae4', '#0ddafa', '#9cb1fe', '#bb9cfe', '#e59cfe',
                  '#fe9cd6', '#fc9e93', '#fe9cd6', '#e59cfe', '#bb9cfe', '#9cb1fe', '#0ddafa', '#0dfae4', '#0dfa9d',
                  '#1ffa0d', '#8bfa0d']
        colors = colors + colors + colors + colors
        trajectory_pts = np.linspace(0, np.pi, len(colors)+1)
        next_pts = [SE3Animation.__trajectory(t) for pt_index, t in enumerate(trajectory_pts)]

        points = np.stack([self.coordinates[0].ravel(),
                           self.coordinates[1].ravel(),
                           self.coordinates[2].ravel(),
                           np.ones(self.coordinates[0].size)])
        geo_lines = SE3Animation.__sphere_geo_lines()

        def update(frame: int) -> None:
            self.ax.clear()
            self.__reset_axis()
            self.__animation_step(next_pts[frame], frame)
            self.__draw_trajectory(next_pts, frame)
            T = self.transform(self.next_step)
            self.__draw_next_sphere(colors[frame], geo_lines, points, T)

        self.__reset_axis()
        self.ax.plot([next_pts[0][0], np.array([0.0])],
                     [next_pts[0][1], np.array([0.0])],
                     [next_pts[0][2], np.array([0.0])],
                     color='red',
                     linewidth=4,
                     label='Line between points')

        self.__draw_sphere( '#facf0d', geo_lines)
        ani = FuncAnimation(self.fig, update, frames=len(colors), interval=self.interval, repeat=False, blit=False)
        # plt.show()
        ani.save('SE3_animation.mp4', writer='ffmpeg', fps=self.fps, dpi=240)

    """ ------------------  Private Helper Methods -------------------  """

    def __animation_step(self, next_point: np.array, frame: int) -> None:
        self.next_step = [frame*0.1, next_point]

    def __draw_trajectory(self, next_pts: np.array, frame: int) -> None:
        self.ax.plot([next_pts[0][0], np.array([0.0])],
                     [next_pts[0][1], np.array([0.0])],
                     [next_pts[0][2], np.array([0.0])],
                     color='red',
                     linewidth=3,
                     label='Line between points')
        for idx in range(frame + 1):
            next_pt = next_pts[idx + 1]
            pt = next_pts[idx]
            self.ax.plot([next_pt[0], pt[0]],
                         [next_pt[1], pt[1]],
                         [next_pt[2], pt[2]],
                         color='red',
                         linewidth=3,
                         label='Line between points')

    @staticmethod
    def __trajectory(t: np.array) -> np.array:
        import math
        radius = 1.4
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        u = (t/np.pi - 1/2)
        z = 10*math.exp(-u*u) - np.array(9)
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
        # return self.__draw_dot()

    def __draw_next_sphere(self, color: AnyStr, geo_lines: List[np.array], points: np.array, T):
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
                    linewidth = 4
                case 15:
                    color = 'black'
                    linewidth = 4
                case _:
                    color = 'black'
                    linewidth = 1
            self.ax.plot(transformed_line[0],
                         transformed_line[1],
                         transformed_line[2],
                         color=color,
                         linewidth=linewidth,
                         alpha=1.0)

    @staticmethod
    def __sphere_geo_lines() -> List[np.array]:
        latitudes = np.linspace(-np.pi / 2, np.pi / 2, 9)
        longitudes = np.linspace(0, 2 * np.pi, 18)

        geo_lines = []
        for lat in latitudes:
            phi = np.linspace(0, 2 * np.pi, 100)
            x_lat = np.cos(phi) * np.cos(lat)
            y_lat = np.sin(phi) * np.cos(lat)
            z_lat = np.ones_like(phi) * np.sin(lat)
            geo_lines.append(np.stack([x_lat, y_lat, z_lat, np.ones_like(phi)]))

        for lon in longitudes:
            theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
            x_lon = np.cos(theta) * np.cos(lon)
            y_lon = np.cos(theta) * np.sin(lon)
            z_lon = np.sin(theta)
            geo_lines.append(np.stack([x_lon, y_lon, z_lon, np.ones_like(theta)]))
        return geo_lines

    def __draw_dot(self) -> np.array:
        lat = np.radians(45)
        lon = np.radians(-60)
        x_dot = np.cos(lat) * np.cos(lon)
        y_dot = np.cos(lat) * np.sin(lon)
        z_dot = np.sin(lat)
        self.ax.scatter(x_dot,
                        y_dot,
                        z_dot,
                        color='black',
                        s=150,
                        label='Original Point',
                        alpha=1.0,
                        zorder=10,
                        depthshade=False)
        return np.array([[x_dot], [y_dot], [z_dot], [1.0]])

    def __reset_axis(self):
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlim(-1.7, 1.7)
        self.ax.set_ylim(-1.7, 1.7)
        self.ax.set_zlim(-1.7, 1.7)
        self.ax.set_title("SE(3) Transformation on a 3D Sphere",
                          fontdict={'fontsize': 19, 'fontweight': 'bold', 'fontname': 'Helvetica', 'color': 'black'})
        self.ax.set_xlabel('X', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        self.ax.set_ylabel('Y', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        self.ax.set_zlabel('Z', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        self.__display_mathtex()

    def __display_mathtex(self):
        formula1 = r"$ cos(\theta) \ -sin(\theta) \ \ 0 \ \ t_{1}$"
        formula2 = r"$sin(\theta) \ \ \ \ \ cos(\theta) \ \ 0 \ \ t_{2}$"
        formula3 = r"$ \ \ \  0 \ \ \ \ \ \ \ \ \ \ \ 0 \ \ \ \ \ \ \ 1 \ \ t_{3}$"
        formula4 = r"$ \ \ \  0 \ \ \ \ \ \ \ \ \ \ \ 0 \ \ \ \ \ \ \  0 \ \ 1 \ $"
        formulas = [formula1, formula2, formula3, formula4]
        top_z = 1.45
        for idx in range(len(formulas)):
            self.ax.text(x=-3.8,
                         y=1.2,
                         z=top_z - 0.3 * idx,
                         s=formulas[idx],
                         horizontalalignment='left',
                         fontdict={'fontsize': 13, 'fontweight': 'bold', 'color': 'black'},
                         bbox=dict(facecolor='lightblue', edgecolor='lightblue'))
        self.ax.plot([-3.9, -3.9],
                     [1.2, 1.2],
                     [3.0, 0.5],
                     color='black',
                     linewidth=1)
        self.ax.plot([-3.94, -3.94],
                     [1.2, 1.2],
                     [3.0, 0.5],
                     color='black',
                     linewidth=1)
        self.ax.plot([-1.66, -1.66],
                     [1.2, 1.2],
                     [3.0, 1.1],
                     color='black',
                     linewidth=1)
        self.ax.plot([-1.62, -1.62],
                     [1.2, 1.2],
                     [3.0, 1.1],
                     color='black',
                     linewidth=1)


if __name__ == '__main__':
    def lie_transform(args: List[np.array]) -> np.array:
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

    lie_group_simulation = SE3Animation.build(transform=lie_transform, interval=2000, fps=10)
    lie_group_simulation.draw()



