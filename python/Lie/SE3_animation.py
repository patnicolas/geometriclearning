__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import numpy as np
from typing import AnyStr, List, Self, Callable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from util.base_animation import BaseAnimation
import logging
logger = logging.getLogger('Lie.SE3Animation')
__all__ = ['SE3Animation']


class SE3Animation(BaseAnimation):
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
                 chart_pos: List[float],
                 interval: int,
                 fps: int,
                 coordinates: (float, float, float),
                 SE3_transform: Callable[[np.array], np.array]) -> None:
        """
        Default constructor for the animation of SE3 lie Group
        @param chart_pos: Define the position of the chart [x, y, width, height]
        @type chart_pos: List[float]
        @param interval: Interval in milliseconds between frames
        @type interval: int
        @param fps: Number of frame per seconds for animation
        @type fps: int
        @param coordinates: Initial coordinate of the sphere used for SE3 transformation
        @type coordinates: Tuple[float, float,float]
        @param SE3_transform: Rotation (SO3) + Translation transform
        @type SE3_transform: Callable
        """
        super(SE3Animation, self).__init__(chart_pos, interval, fps)

        self.coordinates = coordinates
        self.transform = SE3_transform
        self.next_step = [np.array(0.0), np.array([[0.0], [0.0], [0.0]])]
        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')

    @classmethod
    def build(cls,
              chart_pos: List[float],
              interval: int,
              fps: int,
              transform: Callable[[np.array], np.array]) -> Self:
        """
        Alternative constructor that takes a SE3 transformation as argument
        @param chart_pos: Define the position of the chart [x, y, width, height]
        @type chart_pos: List[float]
        @param interval: Interval in milliseconds between frames
        @type interval: int
        @param fps: Number of frame per seconds for animation
        @type fps: int
        @param transform: Rotation (SO3) + Translation transform
        @type transform: Callable
        @return: Instance of SE3Animation
        @rtype: SE3Animation
        """
        u, v = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        return cls(chart_pos=chart_pos, interval=interval, fps=fps, coordinates=(x, y, z), SE3_transform=transform)

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
        self.fig.patch.set_facecolor('#f0f9ff')
        self.ax.set_facecolor('#f0f9ff')
        self._draw_logo(fig=self.fig)
        self.__draw_formula()

        geo_lines = SE3Animation.__sphere_geo_lines()

        def update(frame: int) -> None:
            """
            Update method to be executed for each frame
            @param frame: Number of the frame (index) used in the simulation
            @type frame: int
            """
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

    def __draw_formula(self) -> None:
        import matplotlib.image as mpimg
        img = mpimg.imread('../input/SE3_formula.png')
        inset_ax = self.fig.add_axes([0.01, 0.32, 0.26, 0.26])
        inset_ax.imshow(img, alpha=1.0)
        inset_ax.axis('off')

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
        radius = 1.5
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

    def __reset_axis(self):
        self.ax.set_box_aspect([1.3, 1.3, 1.2])
        self.ax.set_xlim(-1.8, 1.8)
        self.ax.set_ylim(-1.8, 1.8)
        self.ax.set_zlim(-1.8, 1.8)
        self.ax.set_title(x=0.5,
                          y=1.0,
                          label="SE(3) Transformation on a 3D Sphere",
                          fontdict={'fontsize': 21, 'fontweight': 'bold', 'fontname': 'Helvetica', 'color': 'black'})
        self.ax.set_xlabel('X', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        self.ax.set_ylabel('Y', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        self.ax.set_zlabel('Z', fontdict={'fontsize': 14, 'fontweight': 'bold'})


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


    """
    from PIL import Image

    # Load image
    img = Image.open('../input/Background_color.png')

    # Convert to RGB if needed
    img = img.convert('RGB')

    # Get the RGB value of a pixel (x=10, y=20 for example)
    r, g, b = img.getpixel((10, 20))

    # Convert to HEX
    hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
    print("Hex color:", hex_color)

    """
    lie_group_simulation = SE3Animation.build(chart_pos=[-0.4, -0.1, 2.2, 1.1],
                                              interval=2000,
                                              fps=10,
                                              transform=lie_transform,)
    lie_group_simulation.draw()



