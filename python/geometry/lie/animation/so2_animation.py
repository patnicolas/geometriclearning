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
from typing import List, Callable, AnyStr, Dict, Any
# 3rd Party imports
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
# Library imports
from util.base_animation import BaseAnimation
__all__ = ['SO2Animation']


class SO2Animation(BaseAnimation):
    """
        Dictionary of animation configuration parameters
    ------------------------------------------------
    logo_pos: Tuple[int, int]   Position of the logo if one is defined
    logo_size: Tuple[int, int]  Size of the logo if one is defined
    interval: int  Interval for FuncAnimation in msec
    fps: int  Frame per second
    sphere_radius: float  Radius of sphere in 3D space
    x_lim: Tuple[float, float]  Range of x values
    y_lim: Tuple[float, float]  Range of y values
    formula_pos: Tuple[float, float]  Position of formula if any
    title_pos: Tuple[float, float]  Position of title
    num_frames: int  Number of frames in the animation
    """
    def __init__(self, rotation_steps: List[Callable[[int], float]], **kwargs: Dict[AnyStr, Any]) -> None:
        super(SO2Animation, self).__init__(**kwargs)
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        _arrow1 = ax.plot([], [], 'ro-', linewidth=2, color='red')
        _arrow2 = ax.plot([], [], 'ro-', linewidth=2, color='blue')
        _arrow3 = ax.plot([], [], 'ro-', linewidth=2, color='green')
        _arrow4 = ax.plot([], [], 'ro-', linewidth=2, color='black')
        _arrow5 = ax.plot([], [], 'ro-', linewidth=2, color='purple')
        self.arrows = [_arrow1[0], _arrow2[0], _arrow3[0], _arrow4[0], _arrow5[0]]
        self.models = rotation_steps

    @staticmethod
    def rot2d(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def _group_name(self) -> AnyStr:
        return 'SO2'

    def draw(self, mp4_file: bool = False) -> None:
        """
            Draw and animate a 2D circle in an ambient Euclidean plane. The animation is driven by Matplotlib
            FuncAnimation class that require an update nested function.

            @param mp4_file: Flag to specify if the mp4 file is to be generated (False plot are displayed but not saved)
            @type mp4_file: boolean
        """
        from matplotlib.patches import Circle

        self.fig.patch.set_facecolor('#f0f9ff')
        self.ax.set_facecolor('#f0f9ff')
        x_lim = self.config['x_lim']
        y_lim = self.config['y_lim']
        self.ax.set_xlim(x_lim[0], x_lim[1])
        self.ax.set_ylim(y_lim[0], y_lim[1])
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        circle = Circle((0.0, 0.0), 1.0, edgecolor='blue', facecolor='lightblue', linewidth=2)
        self.ax.add_patch(circle)
        self.ax.set_title(y=self.config['title_pos'][0],
                          x=self.config['title_pos'][1],
                          label="SO(2) Rotation",
                          fontdict={'fontsize': 16, 'fontname': 'Helvetica'})
        self._draw_logo(fig=self.config['fig_size'])

        # Update function for animation
        def update(frame: int) -> None:
            for index in range(len(self.arrows)):
                frame = self.models[index](frame)
                theta = frame * np.pi / 2400
                R = SO2Animation.rot2d(theta)
                vec = R @ np.array([1, 0])  # Rotate unit vector
                self.arrows[index].set_data([0, vec[0]], [0, vec[1]])

        # Create animation
        ani = FuncAnimation(self.fig,
                            update,
                            frames=self.config['num_frames'],
                            repeat=False,
                            interval=self.config['interval'],
                            blit=False)
        if mp4_file:
            fps = self.config['fps']
            ani.save(filename=f'{self._group_name()}_animation.mp4', writer='ffmpeg', fps=fps, dpi=240)
        else:
            plt.show()


if __name__ == '__main__':
    rot_steps = [
        lambda a: 0.03 * a * a + a + 2,
        lambda a: 0.04 * a * a + a + 3,
        lambda a: 0.05 * a * a,
        lambda a: 0.012 * a * a - a,
        lambda a: 0.005 * a * a - a
    ]
    config = {
        'fig_size': (10, 8),
        'logo_pos': (0.1, 0.97),
        'title_pos': (0.6, 1.0),
        'x_lim': (-1.8, 1.8),
        'y_lim': (-1.8, 1.8),
        'sphere_radius': 1.2,
        'interval': 1000,
        'fps': 8,
        'num_frames': 40
    }
    so2_animation = SO2Animation(rotation_steps=rot_steps, kwargs=config)
    so2_animation.draw(mp4_file=True)
