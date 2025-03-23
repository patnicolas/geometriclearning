from manim import *
import numpy as np

class SE3LieGroupRotation(ThreeDScene):
    def construct(self):
        # Set up camera
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)

        # Axes and sphere
        axes = ThreeDAxes(x_range=[-1, 7], y_range=[-4, 4], z_range=[-4, 4])

        sphere = Surface(
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

        # Create a group to apply SE(3) transformation
        se3_object = VGroup(sphere)
        self.add(axes, se3_object)

        # Add a title
        title = Text("SE(3) Transformation", font_size=36)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        self.wait(1)

        # Rotation around Y-axis (SO(3) part)
        self.play(Rotate(se3_object, angle=PI / 2, axis=UP, about_point=ORIGIN, run_time=2))
        self.wait(0.5)

        # Translation along X-axis [5, 0, 0] (R^3 part)
        self.play(se3_object.animate.shift(RIGHT * 5), run_time=2)
        self.wait(1)

        # Display final state
        label = Text("Rotation + Translation = SE(3)", font_size=28)
        label.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(label)
        self.play(FadeIn(label))
        self.wait(2)
