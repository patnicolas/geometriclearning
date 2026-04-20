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
from typing import Tuple, Callable, Any, List
from dataclasses import dataclass

@dataclass
class GaugeConfig:
    radius: float
    num_ticks: int
    font_size: int


class GaugeGroup(VGroup):

    def __init__(self,
                 vt: ValueTracker,
                 gauge_config: GaugeConfig,
                 data_points: List[float],
                 **kwargs) -> None:
        super(GaugeGroup, self).__init__(**kwargs)

        self.vt = vt
        self.radius = gauge_config.radius  # 2.5
        self.num_ticks = gauge_config.num_ticks  # 0 to 10
        self.data_points = data_points

        self.limit_values = (min(data_points), max(data_points))
        ticks, labels = GaugeGroup.create_ticks(gauge_config, self.limit_values)

        gradient_background = AnnularSector(
            inner_radius=0.5,
            outer_radius=gauge_config.radius-0.3,
            start_angle=PI,
            angle=-PI,
            stroke_width=2,
            fill_opacity=0.5,
            color=DARK_GREY
        )  # Manim interpolates colors automatically

        gauge_arc = Arc(radius=gauge_config.radius,
                        start_angle=PI,
                        angle=-PI,
                        stroke_width=4,
                        color=LIGHT_GREY)

        self.needle = Line(ORIGIN, LEFT * 1.8, buff=0, stroke_width=6, color=RED)
        self.needle.add_tip(tip_shape=StealthTip, tip_length=0.2)

        # 4. Add a pivot point at the center
        center_dot = Dot(radius=0.1, color=BLUE)
        self.add(gauge_arc, gradient_background, self.needle, center_dot, ticks, labels)

    def get_updater(self, vt) -> Callable[[Any], Any]:
        def updater(obj):
            idx = int(vt.get_value())
            value = self.data_points[idx]/self.limit_values[1]
            angle = interpolate(PI, 0, value/self.limit_values[0])
            obj.needle.set_angle(angle)
        return updater

    @staticmethod
    def create_ticks(gauge_config: GaugeConfig,
                     limit_values: Tuple[float, float],
                     limit_angles: Tuple[float, float] = (PI, 0)) -> Tuple[VGroup, VGroup]:
        ticks = VGroup()
        labels = VGroup()

        for i in range(gauge_config.num_ticks):
            # Calculate the angle for this specific tick, alpha goes from 0 to 1
            alpha = i / (gauge_config.num_ticks - 1)
            angle = interpolate(limit_angles[0], limit_angles[1], alpha)

            # Polar to Cartesian for tick positioning
            pos_on_arc = np.array([np.cos(angle), np.sin(angle), 0])

            # Create a line pointing toward the center
            tick = Line(
                pos_on_arc * gauge_config.radius,
                pos_on_arc * (gauge_config.radius - 0.2),
                stroke_width=6,
                color=BLUE
            )
            ticks.add(tick)

            # Add numeric labels
            label_val = int(interpolate(limit_values[0], limit_values[1], alpha))
            label = (Text(str(label_val), font_size=gauge_config.font_size)
                     .move_to(pos_on_arc * (gauge_config.radius - 0.6)))
            labels.add(label)
        return ticks, labels


class GaugeScene(Scene):
    radius = 2.5
    num_ticks = 11

    def construct(self):
        import math
        vt = ValueTracker(0)
        gauge_config = GaugeConfig(radius=2.5, num_ticks=11, font_size=16)


        gauge_group = GaugeGroup(vt, gauge_config, [math.sin(0.01*x) for x in range(0, 25)])
        self.add(gauge_group)

        self.play(gauge_group.vt.animate.set_value(75), run_time=2, rate_func=bezier([0, 0, 1, 1]))
        self.wait()
        self.play(gauge_group.vt.animate.set_value(20), run_time=1.5)
        self.wait()


if __name__ == '__main__':
    scene = GaugeScene()
    scene.construct()