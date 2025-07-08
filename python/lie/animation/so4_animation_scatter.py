

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import List
from util.base_animation import BaseAnimation


class SO4AnimationScatter(BaseAnimation):
    def __init__(self,
                 logo_pos: List[float],
                 interval: int,
                 fps: int,
                 num_points: int,
                 sphere_radius: float = 1.0) -> None:
        super(SO4AnimationScatter, self).__init__(logo_pos, interval, fps)
        self.num_points = num_points
        self.fig = plt.figure(figsize=(10, 7))
        self.sphere_radius = sphere_radius
        self.ax = self.fig.add_subplot(111, projection='3d')


    def draw(self, mp4_file: bool = False) -> None:
        points_on_sphere = SO4AnimationScatter.__generate_4d_sphere(self.num_points)

        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_box_aspect([1, 1, 1])

        # Animation function
        def update(frame: int) -> None:
            theta = np.radians(frame)
            R = SO4AnimationScatter.__so4_rotation_matrix(theta, 2 * theta)
            rotated = points_on_sphere @ R.T
            proj = SO4AnimationScatter.__project_rotation(rotated)
            Z = proj[:, 2]
            self.ax.plot_wireframe(X=proj[:, 0], Y=proj[:, 1], Z=proj[:, 2], s=8, color='C0', cmap='viridis_r')

        ani = FuncAnimation(self.fig,
                            update,
                            frames=np.linspace(0, 360, 120),
                            repeat=False,
                            interval=50)
        plt.show()

    @staticmethod
    def __project_rotation(points: np.array) -> np.array:
        return points[:, :3]

    @staticmethod
    def __generate_4d_sphere(num_points: int) -> np.array:
        x = np.random.normal(size=(4, num_points))
        x /= np.linalg.norm(x, axis=0)
        return x.T  # shape: (num_points, 4)

    # Rotation in 4D: composed of two SO(2) rotations
    @staticmethod
    def __so4_rotation_matrix(theta1: np.array, theta2: np.array) -> np.array:
        R = np.eye(4)
        R[:2, :2] = [[np.cos(theta1), -np.sin(theta1)],
                     [np.sin(theta1),  np.cos(theta1)]]
        R[2:, 2:] = [[np.cos(theta2), -np.sin(theta2)],
                     [np.sin(theta2),  np.cos(theta2)]]
        return R


if __name__ == '__main__':
    # so4_animation = SO4Animation(chart_pos=[-0.4, -0.1, 2.2, 1.1], interval=50, fps=20, num_points=1000)
    # so4_animation.draw()

    # Generate points on a 4D spherical grid (theta1, theta2, phi)
    def generate_4d_sphere_grid(n_theta=70, n_phi=120):
        theta = np.linspace(0, np.pi, n_theta)  # polar angle for 3D slice
        phi = np.linspace(0, 2 * np.pi, n_phi)  # azimuthal angle
        theta1, theta2 = np.meshgrid(theta, phi)

        x = np.cos(theta1)
        y = np.sin(theta1) * np.cos(theta2)
        z = np.sin(theta1) * np.sin(theta2)
        w = np.zeros_like(z)

        # Reshape to (num_points, 4)
        sphere_4d = np.stack([x, y, z, w], axis=-1)
        return sphere_4d, x.shape


    # SO(4) rotation matrix
    def so4_rotation_matrix(theta1, theta2):
        R = np.eye(4)
        R[[0, 0, 1, 1], [0, 1, 0, 1]] = [np.cos(theta1), -np.sin(theta1), np.sin(theta1), np.cos(theta1)]
        R[[2, 2, 3, 3], [2, 3, 2, 3]] = [np.cos(theta2), -np.sin(theta2), np.sin(theta2), np.cos(theta2)]
        return R


    # Project 4D to 3D
    def project_to_3d(points_4d):
        return points_4d[..., :3]  # ignore w component


    # Initialization
    sphere_4d, shape_2d = generate_4d_sphere_grid()
    sphere_4d_flat = sphere_4d.reshape(-1, 4)

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1, 1, 1])

    # Create line containers
    wire_u = [ax.plot([], [], [], 'blue', lw=0.5)[0] for _ in range(shape_2d[0])]
    wire_v = [ax.plot([], [], [], 'red', lw=0.5)[0] for _ in range(shape_2d[1])]


    # Animation function
    def update(frame):
        theta = np.radians(frame)
        R = so4_rotation_matrix(theta, 2 * theta)
        rotated = sphere_4d_flat @ R.T
        projected = project_to_3d(rotated).reshape(*shape_2d, 3)

        # Draw constant-u lines (theta1)
        for i in range(shape_2d[0]):
            x = projected[i, :, 0]
            y = projected[i, :, 1]
            z = projected[i, :, 2]
            wire_u[i].set_data(x, y)
            wire_u[i].set_3d_properties(z)

        # Draw constant-v lines (theta2)
        for j in range(shape_2d[1]):
            x = projected[:, j, 0]
            y = projected[:, j, 1]
            z = projected[:, j, 2]
            wire_v[j].set_data(x, y)
            wire_v[j].set_3d_properties(z)

        return wire_u + wire_v


    ani = FuncAnimation(fig, update, frames=np.linspace(0, 360, 400), interval=50)
    # plt.show()
    ani.save('SO4_b_animation.mp4', writer='ffmpeg', fps=12, dpi=240)
