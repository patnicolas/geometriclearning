from animation import *
import numpy as np


class SE3LieGroupRotation(ThreeDScene):
    def construct(self):
        # Set up camera
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)

        y_label = MathTex(
            r"Rotation \ Y \ Axis  \ \ \begin{bmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ -1 & 0 & 0 \end{bmatrix}")
        x_translation = MathTex(
            r"Translation \ X \ Axis  \ \ [4, 0, 0]")
        z_translation = MathTex(
            r"Translation \ X \ Axis  \ \ [0, 0, 1]")
        z_label = MathTex(
            r"Rotation \ Y \ Axis  \ \ \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}")
        eq_label = MathTex(r"Lie \ Group: \ \ SE(3)=\left\{ \begin{vmatrix} R & t \\ 0 & 1 \end{vmatrix} \in \mathbb{R}^{4 \ast 4} \ \ | \ R \in SO(3), t \in \mathbb{R}^{3} \right\}")
        y_label.to_corner(UL).scale(0.5)
        x_translation.to_corner(UL).scale(0.5)
        z_translation.to_corner(UL).scale(0.5)
        z_label.to_corner(UL).scale(0.5)
        eq_label.to_edge(DOWN).scale(0.5)

        # Axes and sphere
        axes = ThreeDAxes(x_range=[-2, 6], y_range=[-4, 4], z_range=[-4, 4])

        red_sphere = Surface(
            lambda u, v: np.array([
                np.cos(u) * np.sin(v),
                np.sin(u) * np.sin(v),
                np.cos(v)
            ]),
            u_range=[0, 2 * PI],
            v_range=[0, PI],
            resolution=(50, 50),
            fill_opacity=0.7,
            checkerboard_colors=[RED_E, RED_D],
        )
        blue_sphere = Surface(
            lambda u, v: np.array([
                np.cos(u) * np.sin(v),
                np.sin(u) * np.sin(v),
                np.cos(v)
            ]),
            u_range=[0, 2 * PI],
            v_range=[0, PI],
            resolution=(50, 50),
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_E, BLUE_C],
        )

        # Create a group to apply SE(3) transformation
        se3_object = VGroup(red_sphere)
        self.add(axes, se3_object)
        # Add the equation
        self.add_fixed_in_frame_mobjects(eq_label)
        self.play(Write(eq_label))

        self.wait(0.5)

        self.add_fixed_in_frame_mobjects(y_label)
        # Rotation around Y-axis (SO(3) part)
        self.play(Rotate(se3_object, angle=2*PI, axis=UP, about_point=ORIGIN, run_time=3))
        self.play(Unwrite(y_label))
        self.wait(1.0)

        # Translation along X-axis [4, 0, 0] (R^3 part)
        self.add_fixed_in_frame_mobjects(x_translation)
        self.wait(0.2)
        self.play(se3_object.animate.shift(RIGHT * 4), run_time=4)
        self.play(Unwrite(x_translation))
        self.wait(1)

        se3_object_2 = VGroup(blue_sphere)
        se3_object_2.move_to(se3_object)
        self.add(se3_object_2)
        self.play(FadeOut(red_sphere), FadeIn(blue_sphere))

        # Translation along X-axis [0, 0, 1] (R^3 part)
        self.add_fixed_in_frame_mobjects(z_translation)
        self.wait(0.2)
        self.play(se3_object_2.animate.shift(OUT), run_time=2)
        self.play(Unwrite(z_translation))
        self.wait(1)

        # Rotation around Y-axis (SO(3) part)
        self.add_fixed_in_frame_mobjects(z_label)
        self.wait(0.2)
        self.play(Rotate(se3_object_2, angle=2 * PI, axis=UP, about_point=blue_sphere.get_center(), run_time=4))
        self.play(Unwrite(z_label))
        self.wait(1.0)
