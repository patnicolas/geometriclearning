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
from typing import List, Tuple, AnyStr
from animation.library import get_2d_ranges, colors
from legend_group import LegendGroup, LegendType


class Scatter2DGroup(VGroup):

    def __init__(self,
                 xy_labels: Tuple[AnyStr, AnyStr],
                 lengths: Tuple[int, int],
                 title: MathTex,
                 legend_group: LegendGroup,
                 radius: float,
                 num_lines: int,
                 all_points: List[List[Tuple[float, ...]]],
                 **kwargs) -> None:
        super(Scatter2DGroup, self).__init__(**kwargs)

        data = [item for sublist in all_points for item in sublist]
        xy_ranges = get_2d_ranges(list(zip(*data)), num_lines)

        self.ax = NumberPlane(x_range=xy_ranges[0],
                              y_range=xy_ranges[1],
                              x_length=lengths[0],
                              y_length=lengths[1],
                              background_line_style={
                                  "stroke_color": DARK_GREY,
                                  "stroke_width": 3,
                                  "stroke_opacity": 0.6
                              },
                              x_axis_config={"label_direction": DOWN, "line_to_number_buff": 0.15},
                              y_axis_config={"label_direction": LEFT, "line_to_number_buff": 0.15},
                              axis_config={
                                  "font_size": 18,
                                  "include_ticks": True,
                                  "include_numbers": True,
                                  "decimal_number_config": {"color": LIGHT_GREY},
                              })

        dots = [Dot(point=self.ax.c2p(x, y), color=colors[idx], radius=radius)
                for idx, points in enumerate(all_points) for x, y in points]
        self.dots = VGroup(*dots)

        self.labels = self.ax.get_axis_labels(x_label=MathTex(xy_labels[0], font_size=32),
                                              y_label=MathTex(xy_labels[1], font_size=32))
        self.title = title.next_to(self.dots, UP, buff=0.5)
        self.legends_group = legend_group
        self.add(self.ax, self.labels, self.labels, self.title, self.legends_group, self.dots)

    def get_dynamic(self) -> Create:
        return Create(self.dots)


class Scatter2DScene(Scene):
    data_points = [[(1, 14), (2, 6), (3, 1), (4, 3)], [(1, 2), (0, 4), (3, 9), (4, 3)]]
    legend_texts = ['A', 'B']
    radius = 0.08
    lengths = (3, 3)
    title = "My \ scatter \ plot"
    run_time = 2
    legend_arrange = LEFT
    legend_buff = 0.4
    num_lines = 5

    def construct(self) -> None:
        vt = ValueTracker(0)
        legend_labels = [MathTex(rf"{legend}", font_size=18) for legend in Scatter2DScene.legend_texts]
        legend_group = LegendGroup(legend_labels=legend_labels,
                                   legend_type=LegendType.DOT,
                                   radius=Scatter2DScene.radius,
                                   arrange=Scatter2DScene.legend_arrange,
                                   buff=Scatter2DScene.legend_buff)
        scatter_2d_group = Scatter2DGroup(vt=vt,
                                          xy_labels=("x", "y"),
                                          lengths=Scatter2DScene.lengths,
                                          legend_group=legend_group,
                                          title=MathTex(rf"{Scatter2DScene.title}", font_size=36).to_edge(UP),
                                          radius=Scatter2DScene.radius,
                                          num_lines=Scatter2DScene.num_lines,
                                          xy_ranges=([0.0, 1.0, 10], [0.0, 10.0, 5])).to_edge(LEFT)
        legend_group.next_to(scatter_2d_group, DOWN, buff=0.2)
        box = SurroundingRectangle(scatter_2d_group,
                                   color=DARK_GREY,
                                   buff=0.1,
                                   fill_opacity=0.2,
                                   fill_color=BLACK)
        scatter = scatter_2d_group.get_dynamic()
        self.add(box)
        self.add(scatter_2d_group)
        self.play(scatter, run_time=Scatter2DScene.run_time)


if __name__ == '__main__':
    scene = Scatter2DScene()
    scene.construct()
