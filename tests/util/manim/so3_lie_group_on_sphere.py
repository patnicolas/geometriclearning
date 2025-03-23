from manim import *
import numpy as np


class SO3LieGroupOnSphere(ThreeDScene):
    def construct(self):
        # Set up axes and sphere
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        axes = ThreeDAxes()
        sphere = Surface(
            lambda u, v: np.array([
                np.cos(u) * np.sin(v),
                np.sin(u) * np.sin(v),
                np.cos(v)
            ]),
            u_range=[0, 2 * PI],
            v_range=[0, PI],
            resolution=(30, 30),
            fill_opacity=0.5,
            checkerboard_colors=[PURPLE_E, PURPLE_E]
        )

        self.add(axes, sphere)

        # Axis labels and Lie algebra matrices
        x_label = MathTex(r"Rotation \ X \ Axis \ \ \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{bmatrix}")
        y_label = MathTex(r"Rotation \ Y \ Axis  \ \ \begin{bmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ -1 & 0 & 0 \end{bmatrix}")
        z_label = MathTex(r"Rotation \ Z \ Axis \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}")
        eq_label = MathTex(r"Lie \ Group: \ \  SO(3)=\left\{ A \in \mathbb{R}^{3 \ast 3} \ | \ R.R^{-1}=I_{d}, \    \ det(R)=1 \right\}")
        composition_label = MathTex(r"Composition \ Rotations")

        x_label.to_corner(UL).scale(0.5)
        y_label.to_corner(UL).scale(0.5)
        z_label.to_corner(UL).scale(0.5)
        eq_label.to_corner(RIGHT + UP).scale(0.5)
        composition_label.to_corner(DL)

        self.add_fixed_in_frame_mobjects(eq_label)
        self.wait(0.5)

        # Rotation around X
        self.add_fixed_in_frame_mobjects(x_label)
        self.wait(0.5)
        self.play(Rotate(sphere, angle=PI, axis=RIGHT, run_time=2))
        self.wait(0.5)
        self.play(Unwrite(x_label))
        self.wait(1)
        # Rotation around Y
        self.add_fixed_in_frame_mobjects(y_label)
        self.wait(0.5)
        self.play(Rotate(sphere, angle=PI, axis=UP, run_time=2))
        self.wait(0.5)
        self.play(Unwrite(y_label))
        self.wait(1)
        # Rotation around Z
        self.add_fixed_in_frame_mobjects(z_label)
        self.wait(0.5)
        self.play(Rotate(sphere, angle=PI, axis=OUT, run_time=2))
        self.wait(0.5)
        self.play(Unwrite(z_label))
        self.wait(1)

        # Compose rotations (Lie group composition)
        self.add_fixed_in_frame_mobjects(composition_label)
        self.wait(0.5)
        composite_rotation = Rotate(sphere, angle=PI / 2, axis=RIGHT)
        self.play(composite_rotation)
        self.play(Rotate(sphere, angle=PI / 2, axis=UP))
        self.play(Rotate(sphere, angle=PI / 2, axis=OUT))
        self.wait(1)
        self.move_camera(phi=60 * DEGREES, theta=-90 * DEGREES, run_time=2)
        self.wait(1)
