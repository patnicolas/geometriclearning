__author__ = "Patrick Nicolas"
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
from manim import *


class SphereTangentPlane(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        geodesic_label = MathTex(r"\begin{matrix} Geodesic & \\ x(t)= & cos(t).sin(t) \\ y(t)= & sin(t)^{2} \\ z(t)= & cos(t) \end{matrix}")
        tangent_label = MathTex(r"\begin{matrix}  Tangent & Vector \\ x'(t)= & cos(2t) \\ y'(t)= & sin(2t) \\ z'(t)= & -sin(t) \end{matrix}")
        title = MathTex(r"Manifold \ and \ Tangent \ Space")
        title_2 = MathTex(r"SO3 \ Rotation")
        geodesic_label.to_corner(UL).scale(0.6)
        tangent_label.to_corner(UR).scale(0.6)
        title.to_edge(UP).scale(0.8)
        title_2.to_edge(UP).scale(0.8)

        # Create the sphere
        scale_f = 2.4
        sphere = Sphere(radius=scale_f, resolution=(30, 30))
        sphere.set_fill(PURPLE_E, opacity=0.6)
        self.add(sphere)
        self.add_fixed_in_frame_mobjects(geodesic_label, tangent_label, title)

        # Create functions and derivative
        def geodesic_func(t: np.array) -> np.array:
            return np.array([
                scale_f * np.cos(t) * np.sin(t),
                scale_f * np.sin(t) * np.sin(t),
                scale_f * np.cos(t)
            ])

        def tangent_vec(t: np.array) -> np.array:
            return np.array([np.cos(2*t), np.sin(2*t), -np.sin(t)])

        def normal_vec(t: np.array) -> np.array:
            return np.array([np.sin(t)*np.cos(t), np.sin(t)*np.sin(t), np.cos(t)])

        # Define the parametric function
        start_animation = 0.05
        end_animation = PI - 0.05
        # Define the geodesic: a great circle in the XZ plane
        geodesic = ParametricFunction(
            lambda t: geodesic_func(t),
            t_range=(start_animation, end_animation),
            color=YELLOW
        )
        self.add(geodesic)

        # Object 1: Dot on the geodesic
        moving_dot = Dot3D(geodesic.get_start(), color=RED, radius=0.12)
        self.add(moving_dot)

        # Object 2: Tangent plane setup
        def get_tangent_plane(pos):
            normal = pos / np.linalg.norm(pos)
            tangent_dir = np.cross(normal, np.array([0, 1, 0]))
            if np.linalg.norm(tangent_dir) == 0:
                tangent_dir = np.array([1, 0, 0])
            tangent_dir = tangent_dir / np.linalg.norm(tangent_dir)
            up_dir = np.cross(normal, tangent_dir)
            basis = np.array([tangent_dir, up_dir])
            corners = [
                pos + 0.7 * basis[0] + 0.7 * basis[1],
                pos + 0.7 * basis[0] - 0.7 * basis[1],
                pos - 0.7 * basis[0] - 0.7 * basis[1],
                pos - 0.7 * basis[0] + 0.7 * basis[1],
            ]
            return Polygon(*[p for p in corners], color=WHITE, fill_opacity=0.5)

        tangent_plane = always_redraw(lambda: get_tangent_plane(moving_dot.get_center()))
        self.add(tangent_plane)

        # Object 3: Tangent vector setup (aligned with direction of motion)
        def get_tangent_vector():
            t = tracker.get_value()
            pos = geodesic_func(t)
            direction = tangent_vec(t)
            direction = direction / np.linalg.norm(direction)
            arrow = Arrow3D(
                start=pos,
                end=pos + direction,
                color=RED,
                thickness=0.04
            )
            return arrow

        # Object 4: Normal vector
        def get_normal_vector():
            t = tracker.get_value()
            pos = geodesic_func(t)
            direction = normal_vec(t)
            direction = direction / np.linalg.norm(direction)
            arrow = Arrow3D(
                start=pos,
                end=pos + direction,
                color=BLUE,
                thickness=0.04
            )
            return arrow

        tracker = ValueTracker(start_animation)
        tangent_arrow = always_redraw(get_tangent_vector)
        self.add(tangent_arrow)
        normal_arrow = always_redraw(get_normal_vector)
        self.add(normal_arrow)

        def update_point(mob):
            t = tracker.get_value()
            mob.move_to(geodesic_func(t))

        moving_dot.add_updater(update_point)

        # Animate the dot sliding on the geodesic
        self.play(tracker.animate.set_value(end_animation), run_time=8, rate_func=linear)
        self.play(Unwrite(title))
        self.remove(moving_dot)
        self.play(Unwrite(moving_dot))
        self.add_fixed_in_frame_mobjects(title_2)

        # Rotate the entire group
        bundle = VGroup(sphere, tangent_arrow, normal_arrow, tangent_plane, geodesic)
        self.play(Rotate(bundle, angle=PI, axis=RIGHT, run_time=8))
        self.remove(tangent_plane)
        self.remove(normal_arrow)
        self.remove(tangent_arrow)
        self.wait(1)



