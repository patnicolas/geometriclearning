from util.base_animation import BaseAnimation
from typing import List, Callable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
__all__ = ['SO2Animation']


class SO2Animation(BaseAnimation):
    def __init__(self,
                 logo_pos: List[float],
                 interval: int,
                 fps: int,
                 rotation_steps: List[Callable[[int], float]]) -> None:
        super(SO2Animation, self).__init__(logo_pos, interval, fps)
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
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        circle = Circle((0.0, 0.0), 1.0, edgecolor='blue', facecolor='lightblue', linewidth=2)
        self.ax.add_patch(circle)
        self.ax.set_title(y=1.01, x=0.7, label="SO(2) Rotation", fontdict={'fontsize': 16, 'fontname': 'Helvetica'})
        self._draw_logo(fig=self.fig)

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
                            frames=20,
                            repeat=False,
                            interval=self.interval,
                            blit=False)
        if mp4_file:
            ani.save('SO2_animation.mp4', writer='ffmpeg', fps=self.fps, dpi=240)
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
    so2_animation = SO2Animation(logo_pos=[0.02, 0.785, 0.32, 0.24],
                                 interval=10000,
                                 fps=5,
                                 rotation_steps=rot_steps)
    so2_animation.draw(mp4_file=True)
