__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

from manim import Scene


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
from legend_group import LegendGroup

class Scatter2DGroup(VGroup):

    def __init__(self,
                 x_label: AnyStr,
                 y_label: AnyStr,
                 lengths: Tuple[int, int],
                 title: MathTex,
                 legend_group: LegendGroup,
                 all_points: List[List[Tuple[float, ...]]],
                 **kwargs) -> None:
        super(Scatter2DGroup, self).__init__(**kwargs)

        data = [item for sublist in all_points for item in sublist]
        transposed = zip(*data)
        data = list(transposed)
        x_range, y_range = get_2d_ranges(data)

        self.ax = NumberPlane(x_range=x_range,
                              y_range=y_range,
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
                                  "font_size": 20,
                                  "include_ticks": True,
                                  "include_numbers": True,  # This is often more reliable than .add_coordinates()
                                  "decimal_number_config": {"color": LIGHT_GREY},  # Force a bright color
                              })

        dots = [Dot(point=self.ax.c2p(x, y), color=colors[idx], radius=0.08)
                for idx, points in enumerate(all_points) for x, y in points]
        self.dots = VGroup(*dots)
        self.labels = self.ax.get_axis_labels(x_label=x_label, y_label=y_label)
        self.title = title.next_to(self.dots, UP, buff=0.5)
        self.legends_group = legend_group
        self.add(self.ax, self.labels)

    def get_attributes(self) -> Tuple[Write | Create, ...]:
        return Write(self.title), Write(self.legends_group), Write(self.labels), Create(self.ax), Create(self.dots)


class Scatter2DScene(Scene):

    def construct(self) -> None:
        # data_points = [(0.1*x+0.1*random.random(), 0.35*x-0.05*random.random()) for x in range(10)]
        data_1 = [(1, 2), (2, 6), (3, 1), (4, 3)]
        data_2 = [(1, 2), (0, 4), (-3, 9), (4, 3)]
        data_points = [data_1, data_2]
        # data_points = [(n*0.1, m*0.1) for n, m in data_points]
        legend_group = LegendGroup(legend_labels=[MathTex(r"Set 1", font_size=18),
                                                  MathTex(r"Set 2", font_size=18)])
        scatter_2d_group = Scatter2DGroup(x_label="x",
                                          y_label="y",
                                          lengths=(3, 3),
                                          legend_group=legend_group,
                                          title=MathTex(r" \text{Scatter plot}",
                                                                   font_size=36).to_edge(UP),
                                          all_points=data_points)
        legend_group.next_to(scatter_2d_group, DOWN, buff=0.2)
        title, legend, labels, axes, scatter = scatter_2d_group.get_attributes()
        self.play(title, legend, labels, axes, scatter, run_time=2)


if __name__ == '__main__':
    """
    scale = 0.001
    data_points = [(1, 2), (2, 6), (3, 1), (4, 3)]
    data_points = [(n * scale, m * scale) for n, m in data_points]
    x_range, y_range = get_2d_ranges(list(zip(*data_points)))
    print(x_range)
    """
    scene = Scatter2DScene()
    scene.construct()
