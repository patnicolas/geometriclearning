

from manim import *
import numpy as np

from manim import *
import numpy as np


class HeatPropagation3D(ThreeDScene):
    def construct(self):
        # 1. Physics Parameters
        alpha = 0.15  # Thermal diffusivity
        bar_length = 8
        segments_count = 40
        time = ValueTracker(0.05)  # Start slightly above 0

        # 2. Camera setup
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        # 3. Create the segmented 3D Bar
        bar = VGroup()
        dx = bar_length / segments_count

        for i in range(segments_count):
            # Calculate position along the bar (from -4 to 4)
            x_pos = -4 + (i + 0.5) * dx

            # Create a 3D slice
            segment = Prism(dimensions=[dx, 1, 1])
            segment.move_to([x_pos, 0, 0])
            segment.set_stroke(opacity=0.1)  # Soft edges for better glow effect

            # 4. The Thermal Color Updater
            def update_segment(m, x=x_pos):
                t = time.get_value()
                # Heat source is at the left end: x = -4
                dist_from_source = x + 4

                # Solution to 1D heat equation for a point source
                temp = (1 / np.sqrt(4 * np.pi * alpha * t)) * np.exp(-(dist_from_source ** 2) / (4 * alpha * t))

                # Normalize temp to a 0.0 - 1.0 range for the color map
                # As time increases, max temp drops, so we scale the denominator
                norm_temp = np.clip(temp / (1.2 / np.sqrt(t)), 0, 1)

                m.set_fill(
                    color=interpolate_color(BLUE_E, RED, norm_temp),
                    opacity=0.3 + (norm_temp * 0.6)  # Hotter areas become more opaque
                )

            segment.add_updater(update_segment)
            bar.add(segment)

        # 5. Visual indicators
        source_glow = Dot3D(point=[-4, 0, 0], color=RED, radius=0.2)
        source_glow.add_updater(lambda m: m.set_opacity(1 if time.get_value() < 1 else 0))

        # 6. Layout UI
        self.add_fixed_in_frame_mobjects()
        label = Text("3D Thermal Diffusion", font="Monospace").to_edge(UP)
        time_display = Variable(time.get_value(), Text("Time"), num_decimal_places=2).to_corner(UL)
        time_display.add_updater(lambda v: v.tracker.set_value(time.get_value()))

        # 7. Execution
        self.add(bar, source_glow, label, time_display)
        self.begin_ambient_camera_rotation(rate=0.05)

        # Slow propagation to see the color bleed through the segments
        self.play(
            time.animate.set_value(8.0),
            run_time=15,
            rate_func=linear
        )
        self.stop_ambient_camera_rotation()
        self.wait(2)

class HeatBar(Scene):
    def construct(self):
        # 1. Setup Data for Bar and Heat
        bar_length = 8
        bar_height = 0.5
        num_segments = 50
        alpha = 0.2  # Thermal diffusivity

        # 2. ValueTracker for Time (t)
        time = ValueTracker(0.1)  # Avoid division by zero at t=0

        # 3. Bar Components
        # Represent the bar as a VGroup of many rectangles for smooth gradient
        bar = VGroup(*[
            Rectangle(
                width=bar_length / num_segments,
                height=bar_height,
                stroke_width=1,
                stroke_color=GRAY
            )
            for _ in range(num_segments)
        ]).arrange(RIGHT, buff=0)

        bar_label = Text("Metal Bar (Length=8)").next_to(bar, DOWN, buff=0.5)

        # 4. Heat Distribution Function
        # Fundamental solution (heat kernel) for a point source at x=0
        def heat_kernel(x, t):
            if t <= 0: return 0  # Handle early time
            return (1 / np.sqrt(4 * np.pi * alpha * t)) * np.exp(-(x ** 2) / (4 * alpha * t))

        # 5. Updaters for Thermal Visualization

        # Color Updaters for Bar Segments
        # The map logic must be contained entirely within the lambda function.
        for i, segment in enumerate(bar):
            # Calculate the x-coordinate of the center of this segment,
            # relative to the center of the bar (which is at ORIGIN).
            # The bar goes from x=-4 to x=4.
            segment_x = -4 + (i + 0.5) * (bar_length / num_segments)

            # Map the local variable segment_x to the lambda, but pass 'segment' as the parameter 'm'
            segment.add_updater(
                lambda m, sx=segment_x: m.set_fill(
                    color=interpolate_color(BLUE_E, RED_E, heat_kernel(sx, time.get_value()) / 2),
                    opacity=0.8
                )
            )

        # Heat Source indicator (flashing Dot)
        heat_source = Dot(color=RED_E, radius=0.15).move_to(bar.get_center())

        # 6. Graph View (simultaneous plot)
        ax = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 2, 0.5],
            axis_config={"include_tip": False}
        ).scale(0.8).to_edge(UP, buff=1.5)

        labels = ax.get_axis_labels(x_label="x", y_label="u(x,t)")

        heat_plot = always_redraw(lambda:
                                  ax.plot(lambda x: heat_kernel(x, time.get_value()), color=RED_E, x_range=[-4, 4])
                                  )

        # 7. UI: Time Label
        time_label = Variable(time.get_value(), Text("Time"), num_decimal_places=2)
        time_label.add_updater(lambda v: v.tracker.set_value(time.get_value()))
        time_label.to_corner(UR)

        # 8. Animation Sequence
        self.add(bar, bar_label, heat_source, ax, labels, heat_plot, time_label)
        self.play(Write(time_label))
        self.wait(1)

        # Flash the heat source at t=0
        self.play(Flash(heat_source, color=RED, flash_radius=0.5, num_lines=15, run_time=1))

        # Animate the time, making the heat propagate
        self.play(time.animate.set_value(5.0), run_time=10, rate_func=linear)
        self.wait(3)

"""
class HeatEquation2(Scene):
    def construct(self):
        # 1. Setup Axes
        ax = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 2, 0.5],
            axis_config={"include_tip": False}
        ).scale(0.8)

        labels = ax.get_axis_labels(x_label="x", y_label="u(x,t)")

        # 2. ValueTracker for Time (t)
        # We start at a small t (0.1) to avoid division by zero
        time = ValueTracker(0.1)
        alpha = 0.5  # Thermal diffusivity

        # 3. The Heat Distribution Plot
        # This function solves the heat equation for a point source
        def heat_func(x):
            t = time.get_value()
            return (1 / np.sqrt(4 * np.pi * alpha * t)) * np.exp(-(x ** 2) / (4 * alpha * t))

        # always_redraw ensures the curve updates every frame as 'time' changes
        curve = always_redraw(lambda:
                              ax.plot(heat_func, color=RED, x_range=[-4, 4])
                              )

        # 4. Thermal Color Mapping (Optional Visual Flair)
        # We can add a rectangle that changes color based on the peak temperature
        heat_bar = always_redraw(lambda:
                                 Rectangle(width=8, height=0.2, fill_opacity=0.8)
                                 .set_fill(interpolate_color(BLUE, RED, heat_func(0) / 2))
                                 .next_to(ax, DOWN)
                                 )

        # 5. UI Elements
        time_label = Variable(time.get_value(), Text("Time"), num_decimal_places=2)
        time_label.add_updater(lambda v: v.tracker.set_value(time.get_value()))
        time_label.to_corner(UR)

        # 6. Animation
        self.add(ax, labels, heat_bar)
        self.play(Create(curve), Write(time_label))
        self.wait(1)

        # Animate the heat spreading
        self.play(
            time.animate.set_value(4.0),
            run_time=8,
            rate_func=linear
        )
        self.wait(2)
"""