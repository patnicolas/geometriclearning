__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

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
from typing import Tuple

class GaugeGroup(VGroup):

    def __init__(self, **kwargs) -> None:
        super(GaugeGroup, self).__init__(**kwargs)

        self.vt = ValueTracker(0)

        radius = 2.5
        start_angle = PI
        end_angle = 0
        num_ticks = 11  # 0 to 10

        ticks, labels = GaugeGroup.create_ticks(num_ticks, start_angle, end_angle, radius)

        gradient_background = AnnularSector(
            inner_radius=0.5,
            outer_radius=radius-0.3,
            start_angle=start_angle,
            angle=end_angle-start_angle,
            stroke_width=2,
            fill_opacity=0.5,
            color=DARK_GREY
        )  # Manim interpolates colors automatically

        gauge_arc = Arc(radius=radius,
                        start_angle=start_angle,
                        angle=end_angle-start_angle,
                        stroke_width=4,
                        color=LIGHT_GREY)

        needle = Line(ORIGIN, LEFT * 1.8, buff=0, stroke_width=6, color=RED)
        needle.add_tip(tip_shape=StealthTip, tip_length=0.2)

        # 4. Add a pivot point at the center
        center_dot = Dot(radius=0.1, color=BLUE)
        self.add(gauge_arc, gradient_background, needle, center_dot, ticks, labels)

        needle.add_updater(
            lambda m: m.set_angle(
                interpolate(PI, 0, self.vt.get_value() / 100)
            )
        )

    @staticmethod
    def create_ticks(num_ticks: int, start_angle: float, end_angle: float, radius: float) -> Tuple[VGroup, VGroup]:
        ticks = VGroup()
        labels = VGroup()

        for i in range(num_ticks):
            # Calculate the angle for this specific tick
            # alpha goes from 0 to 1
            alpha = i / (num_ticks - 1)
            angle = interpolate(start_angle, end_angle, alpha)

            # Polar to Cartesian for tick positioning
            pos_on_arc = np.array([np.cos(angle), np.sin(angle), 0])

            # Create a line pointing toward the center
            tick = Line(
                pos_on_arc * radius,
                pos_on_arc * (radius - 0.2),
                stroke_width=6,
                color=BLUE
            )
            ticks.add(tick)

            # Add numeric labels
            label_val = int(interpolate(0, 100, alpha))
            label = Text(str(label_val), font_size=20).move_to(pos_on_arc * (radius - 0.6))
            labels.add(label)
        return ticks, labels


class GaugeScene(Scene):

    def construct(self):
        gauge_group = GaugeGroup()
        self.add(gauge_group)

        self.play(gauge_group.vt.animate.set_value(75), run_time=2, rate_func=bezier([0, 0, 1, 1]))
        self.wait()
        self.play(gauge_group.vt.animate.set_value(20), run_time=1.5)
        self.wait()


if __name__ == '__main__':
    scene = GaugeScene()
    scene.construct()