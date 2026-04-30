
from typing import Callable, List, Tuple

from manim import *
import numpy as np
from math import sin, cos, pi, exp, sqrt
from animation.library.plots.gauge_vgrp import GaugeVGrp, GaugeConfig

class SGDVGRP(VGroup):

    def __init__(self,
                 optimization_surface: Callable[[float, float], List[float]],
                 losses: List[float],
                 noise_level: float,
                 num_steps: int,
                 lr: float,
                 **kwargs) -> None:
        super(SGDVGRP, self).__init__(**kwargs)
        self.optimization_surface = optimization_surface

        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-1, 3, 1],
            axis_config={"include_tip": False}
        )
        labels = axes.get_axis_labels(x_label="x", y_label="y", z_label="Loss")

        # 4. Create the Smooth Surface (the "True" landscape)
        true_surface = Surface(
            lambda x, y: self.optimization_surface(x, y),
            u_range=(-4, 4),  #u_range=(-4, 5),
            v_range=(-4, 4), #v_range=(-4, 5),
            checkerboard_colors=[RED_D, RED_E],
            fill_opacity=0.6,
            stroke_width=0.2
        )

        path_points = SGDVGRP.__get_path_points(optimization_surface,
                                                axes,
                                                num_steps,
                                                noise_level,
                                                lr)
        noisy_path = VGroup()
        for i in range(len(path_points) - 1):
            arrow = Line(
                start=path_points[i],
                end=path_points[i + 1],
                stroke_width=10,
                color=BLUE,
                # Add a subtle glow for "noisy" signal
                stroke_opacity=interpolate(0.3, 1.0, (len(path_points) - i) / len(path_points))
            )
            noisy_path.add(arrow)
        self.noisy_path = noisy_path
        self.path_points = path_points
        self.title = Text("SGD in Latent Space (3D)", color=YELLOW, font_size=32).to_edge(UP, buff=0.4)

        self.noisy_path.set_opacity(0.0)
        # self.add(axes, labels, self.noisy_path, true_surface)
        self.add(axes, true_surface)

    @staticmethod
    def get_data(loss_function: Callable[[float, float], List[float]],
                 num_steps: int,
                 noise_level: float,
                 lr: float) -> List[np.array]:

        def stepping(current_x: float, current_y: float, step: int) -> Tuple[float, float, float]:
            grad_x = current_x
            grad_y = current_y

            # B. The Stochastic "Noise" (Simulating a noisy batch)
            # We add a small random vector to the true gradient
            noise_x = np.random.normal(0, noise_level)
            noise_y = np.random.normal(0, noise_level)

            # C. Noisy Update: w_new = w_old - LR * (Grad + Noise)
            current_x = current_x - lr * (grad_x + noise_x)
            current_y = current_y - lr * (grad_y + noise_y)

            # D. Recalculate Loss at the new point
            current_z = loss_function(current_x, current_y)[2] - 0.052*step+0.5
            return current_x, current_y, current_z

        data_points = []
        curr_x, curr_y = 3.0, 3.0
        for step in range(num_steps):
            curr_x, curr_y, curr_z = stepping(curr_x, curr_y, step)
            data_points.append((curr_x, curr_y, curr_z))
        return data_points

    @staticmethod
    def __get_path_points(loss_function: Callable[[float, float], List[float]],
                          axes: ThreeDAxes,
                          num_steps: int,
                          noise_level: float,
                          lr: float) -> List[np.array]:
        return [axes.c2p(curr_x, curr_y, curr_z) for curr_x, curr_y, curr_z
                in SGDVGRP.get_data(loss_function, num_steps, noise_level, lr)]

def parabolic_loss(x: float, y: float) -> list[float]:
    return [x, y, 0.18 * (x ** 2 + y ** 2)]

def parabolic_trig_loss(x: float, y: float) -> list[float]:
    return [x, y, 0.18 * (x ** 2 + y ** 2) + 0.8 * (cos(x) + sin(y))]

def ackley_loss(x: float, y: float) -> list[float]:
    z = -6.0 * exp(-0.2*sqrt(0.5*(x**2+y**2))) + 0.3*exp(0.5*(cos(0.5*PI*x)+cos(0.5*PI*y))) + 5
    return [x, y, z]

def rosenbrock_loss(x: float, y: float) -> list[float]:
    b = 100.0
    z = (1-x)**3 + b*(y - x**2)**2
    return [x, y, 0.0002*z]

class SGD3DScene(ThreeDScene):

    def construct(self):
        sgd_vgrp = SGDVGRP(optimization_surface=rosenbrock_loss,
                           losses=[],
                           noise_level=0.6,
                           num_steps=40,
                           lr=0.25).scale(0.7).shift(-3.5*OUT).shift(6*LEFT)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.add(sgd_vgrp)
        self.play(Create(sgd_vgrp))

        gauge_config = GaugeConfig(radius=2.5, num_ticks=11, font_size=16)
        gauge_group = GaugeVGrp([sin(0.01 * x) for x in range(0, 25)], gauge_config).shift(3.5 * RIGHT).shift(DOWN)
        self.add_fixed_in_frame_mobjects(sgd_vgrp.title, gauge_group)
        self.wait(2)



        # Set the starting point with a marker
        """
        self.play(Flash(sgd_vgrp.start_marker, color=BLUE))
        self.add(sgd_vgrp.start_marker)
        self.wait(1)
        """

        # self.begin_ambient_camera_rotation(rate=0.04)  # Rotate for depth perspective

        # We animate the path segments one by one
        # sgd_vgrp.noisy_path.set_opacity(1.0)
        # self.play(Create(sgd_vgrp.noisy_path.scale(0.7)), run_time=8)
        # self.stop_ambient_camera_rotation()
        self.wait(2)

    """
    def construct(self):
        # 1. Physics & Learning Parameters
        # A simple saddle function: f(x, y) = x^2 - y^2 (to show noisy descent)
        def loss_function(x: float, y: float) -> list[float]:
            # return np.array([x, y, 0.5 * (x ** 2 + y ** 2)])
            return [x, y, 0.07 * (x ** 2 + y ** 2)]

        # Gradient is [df/dx, df/dy] = [x, y]
        learning_rate = 0.2
        num_steps = 30
        noise_level = 0.8  # Adds "stochastic" randomness to the gradient

        # 2. Configure 3D Camera
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        # 3. Create the 3D Axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-1, 3, 1],
            axis_config={"include_tip": False}
        )

        labels = axes.get_axis_labels(x_label="x", y_label="y", z_label="Loss")

        # 4. Create the Smooth Surface (the "True" landscape)
        true_surface = Surface(
            lambda x, y: loss_function(x, y),
            u_range=(-5, 5),
            v_range=(-5, 5),
            checkerboard_colors=[RED_D, RED_E],
            fill_opacity=0.6,
            stroke_width=0.2
        ).to_edge(DOWN)

        # 5. Stochastic Gradient Descent Loop

        # Start at a fixed point
        curr_x, curr_y = 2.5, 2.5
        curr_z = loss_function(curr_x, curr_y)[2] + 0.8

        # Keep track of the full noisy path
        path_points = [axes.c2p(curr_x, curr_y, curr_z)]

        for _ in range(num_steps):
            # A. Standard Gradient
            grad_x = curr_x
            grad_y = curr_y

            # B. The Stochastic "Noise" (Simulating a noisy batch)
            # We add a small random vector to the true gradient
            noise_x = np.random.normal(0, noise_level)
            noise_y = np.random.normal(0, noise_level)

            # C. Noisy Update: w_new = w_old - LR * (Grad + Noise)
            curr_x = curr_x - learning_rate * (grad_x + noise_x)
            curr_y = curr_y - learning_rate * (grad_y + noise_y)

            # D. Recalculate Loss at the new point
            curr_z = loss_function(curr_x, curr_y)[2] + 0.5
            path_points.append(axes.c2p(curr_x, curr_y, curr_z))

        # 6. Create the Noisy Path Objects
        # We turn our list of points into a VGroup of small arrows
        noisy_path = VGroup()
        for i in range(len(path_points) - 1):
            arrow = Line(
                start=path_points[i],
                end=path_points[i + 1],
                stroke_width=6,
                color=BLUE,
                # Add a subtle glow for "noisy" signal
                stroke_opacity=interpolate(0.3, 1.0, (len(path_points) - i) / len(path_points))
            )
            noisy_path.add(arrow)

        # 7. UI: Layout and Animation
        self.add_fixed_in_frame_mobjects()
        title = Text("SGD in Latent Space (3D)", font_size=36).to_edge(UP)
        legend = VGroup(
            Text("True Loss Landscape", color=BLUE_E),
            Text("Noisy Descent Path (SGD)", color=RED)
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.6).to_edge(LEFT, buff=1)

        self.add(title, legend)

        # 8. Start Animation Sequence
        self.play(Create(axes), Write(labels), Create(true_surface))
        self.wait(1)

        # Set the starting point with a marker
        start_marker = Dot3D(point=path_points[0], color=BLUE, radius=0.1)
        self.play(Flash(start_marker, color=BLUE))
        self.add(start_marker)
        self.wait(1)

        # Slowly draw the wobbly path to emphasize the "descent"
        self.begin_ambient_camera_rotation(rate=0.08)  # Rotate for depth perspective

        # We animate the path segments one by one
        self.play(Create(noisy_path), run_time=8)
        self.stop_ambient_camera_rotation()
        self.wait(2)
    """


if __name__ == '__main__':
    scene = SGD3DScene()
    scene.construct()
