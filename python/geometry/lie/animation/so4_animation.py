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

# Standard Library imports
from typing import Tuple, AnyStr, Any, Dict
# 3rd Party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Library imports
from util.base_animation import BaseAnimation
__all__ = ['SO4Animation']

class SO4Animation(BaseAnimation):
    from mpl_toolkits.mplot3d import Axes3D

    def __init__(self, n_geodesics: Tuple[int, int], **kwargs: Dict[AnyStr, Any]) -> None:
        """
        Default constructor for the SO(4) lie group designed at two 2D rotations

        @param n_geodesics: Number of geodesics for each of the two 2D rotations
        @type n_geodesics: Tuple
        @param **kwargs: Dictionary of configuration parameters for any given animation
        @type **kwargs: Dictionary
        """
        if len(n_geodesics) != 2:
            raise ValueError(f'Number of elements of geodesics {len(n_geodesics)} should be 2')

        super(SO4Animation, self).__init__(**kwargs)
        fig_size = self.config['fig_size']
        self.fig = plt.figure(figsize=fig_size)
        self.n_theta, self.n_phi = n_geodesics
        self.ax = self.fig.add_subplot(111, projection='3d')

    def _group_name(self) -> AnyStr:
        return 'SO4'

    def draw(self, mp4_file: bool = False) -> None:
        """
            Draw and animate a 3D sphere in an ambient Euclidean space. The animation is driven by Matplotlib
            FuncAnimation class that require an update nested function.

            @param mp4_file: Flag of the mp4 file is to be generated (False plot are displayed but not saved)
            @type mp4_file: str
        """
        sphere_4d, shape_theta, shape_phi = self.__create_sphere_geodesics()
        flatten_sphere = sphere_4d.reshape(-1, 4)

        self.fig.patch.set_facecolor('#f0f9ff')
        self.ax.set_facecolor('#f0f9ff')
        self._draw_logo(fig=self.fig)
        self.__draw_formula()

        x_lim = self.config['x_lim']
        y_lim = self.config['y_lim']
        z_lim = self.config['z_lim']
        self.ax.set_xlim(x_lim[0], x_lim[1])
        self.ax.set_ylim(y_lim[0], y_lim[1])
        self.ax.set_zlim(z_lim[0], z_lim[1])
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_title(x=self.config['title_pos'][0],
                          y=self.config['title_pos'][0],
                          label=f"SO4 Transformation - 3D Sphere",
                          fontdict={'fontsize': 18, 'fontweight': 'bold', 'fontname': 'Helvetica', 'color': 'black'})

        # Create wires
        wire_theta = [self.ax.plot([], [], [], 'blue', lw=1.0)[0] for _ in range(shape_theta)]
        wire_phi = [self.ax.plot([], [], [], 'red', lw=1.0)[0] for _ in range(shape_phi)]

        # Callback invoked at each frame
        def update(frame):
            # Compute 4 x4 rotation as a combination of 2 2x2 rotation
            R = SO4Animation.__so4_rotation(frame)
            # Apply the rotation
            rotated = flatten_sphere @ R.T
            # projected = SO4Animation.__project(rotated).reshape(*shape_2d, 3)
            projected = SO4Animation.__project(rotated).reshape(shape_theta, shape_phi, 3)

            # Draw constant-u lines (theta)
            for i in range(shape_theta):
                x = projected[i, :, 0]
                y = projected[i, :, 1]
                z = projected[i, :, 2]
                wire_theta[i].set_data(x, y)
                wire_theta[i].set_3d_properties(z)

            # Draw constant-v lines (phi)
            for j in range(shape_phi):
                x = projected[:, j, 0]
                y = projected[:, j, 1]
                z = projected[:, j, 2]
                wire_phi[j].set_data(x, y)
                wire_phi[j].set_3d_properties(z)
            return wire_theta + wire_phi

        # Trigger the animation.
        ani = FuncAnimation(self.fig,
                            update,
                            frames=np.linspace(0, 360, 300),
                            repeat=False,
                            interval=self.config['interval'])
        # If the animation has to be saved into a MP4 file
        if mp4_file:
            ani.save(f'{self._group_name()}_animation.mp4', writer='ffmpeg', fps=self.config['fps'], dpi=240)
        # Otherwise
        else:
            plt.show()

    """  ----------------   Private supporting methods --------------  """

    def __create_sphere_geodesics(self) -> (np.array, int, int):
        # Prepare variable for the wireframe
        theta = np.linspace(0, np.pi, self.n_theta)
        phi = np.linspace(0, 2 * np.pi, self.n_phi)
        theta1, theta2 = np.meshgrid(theta, phi)

        # 4 coordinates for the wireframe
        x = np.cos(theta1)
        y = np.sin(theta1) * np.cos(theta2)
        z = np.sin(theta1) * np.sin(theta2)
        w = np.zeros_like(z)

        points_on_sphere = np.stack([x, y, z, w], axis=-1)
        return points_on_sphere, x.shape[0], x.shape[1]

    @staticmethod
    def __so4_rotation(frame: int) -> np.array:
        R = np.eye(4)
        # Generate the angles for each of the two 2 x2 rotation
        theta = np.radians(frame)
        phi = 2 * theta
        # Simulate expansion and contraction
        alpha = 1.2 - frame/600
        beta = 1.6 - frame/600
        # Build the 4 dimension rotation as combination of two rotations
        R[[0, 0, 1, 1], [0, 1, 0, 1]] = [alpha*np.cos(phi+np.pi/4),
                                         alpha*np.sin(theta),
                                         -alpha*np.sin(theta),
                                         alpha*np.cos(phi+np.pi/4)]
        R[[2, 2, 3, 3], [2, 3, 2, 3]] = [-beta*np.cos(phi),
                                         -beta*np.sin(theta-np.pi/4),
                                         beta*np.sin(theta-np.pi/4),
                                         -beta*np.cos(phi)]
        return R

    @staticmethod
    def __project(points: np.array) -> np.array:
        return points[..., :3]

    def __draw_formula(self) -> None:
        import matplotlib.image as mpimg

        img = mpimg.imread(f'../../input/{self._group_name()}_formula.png')
        inset_ax = self.fig.add_axes((
            self.config['formula_pos'][0],
            self.config['formula_pos'][1],
            self.config['formula_size'][0],
            self.config['formula_size'][1]
        ))
        inset_ax.imshow(img, alpha=1.0)
        inset_ax.axis('off')


if __name__ == '__main__':
    config = {
        'fig_size': (10, 8),
        'logo_pos': (0.1, 0.71),
        'logo_size': (0.3, 0.3),
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
    so4_animation = SO4Animation(n_geodesics=(60, 160), kwargs=config)
    so4_animation.draw(mp4_file=True)
