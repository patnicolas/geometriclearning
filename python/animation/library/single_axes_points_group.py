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
from typing import Tuple, List, AnyStr
from animation.library import colors
from legend_group import LegendGroup
from animation.library import next_multiple, extract_num_digits
import math


class SingleAxesPointsGroup(VGroup):

    def __init__(self,
                 x_label: AnyStr,
                 y_label: AnyStr,
                 title: MathTex,
                 legend_group: LegendGroup,
                 points: List[Tuple[float, ...]]) -> None:
        super(SingleAxesPointsGroup, self).__init__()

        transposed = zip(*points)
        data = list(transposed)
        x_range, y_range = SingleAxesPointsGroup.__get_ranges(data)

        self.ax = NumberPlane(x_range=x_range,
                              y_range=y_range,
                              x_length=4,
                              y_length=4,
                              background_line_style={
                                  "stroke_color": DARK_GREY,
                                  "stroke_width": 3,
                                  "stroke_opacity": 0.6
                              },
                              x_axis_config={"label_direction": DOWN, "line_to_number_buff": 0.1},
                              y_axis_config={"label_direction": LEFT, "line_to_number_buff": 0.1},
                              axis_config={
                                  "font_size": 29,
                                  "include_ticks": True,
                                  "include_numbers": True,  # This is often more reliable than .add_coordinates()
                                  "decimal_number_config": {"color": LIGHT_GREY},  # Force a bright color
                              })
        self.graphs = [self.ax.plot_line_graph(x_values=data[0],
                                               y_values=data[n+1],
                                               line_color=colors[n],
                                               vertex_dot_style={"color": colors[n]},
                                               name=f"Set {n}") for n in range(len(data)-1)]
        self.labels = self.ax.get_axis_labels(x_label=x_label, y_label=y_label)
        self.title = title.next_to(self.graphs[0], UP, buff=0.3)
        self.legends_group = legend_group
        self.add(title, self.legends_group, self.ax, self.labels, self.graphs)

    def get_attributes(self) -> Tuple[Write | Create, ...]:
        return ((Write(self.title), Write(self.legends_group), Write(self.labels), Create(self.ax)) +
                tuple([Create(graph) for graph in self.graphs]))

    """ ---------------  Private Helper Methods -------------------  """
    @staticmethod
    def __get_ranges(data: List[Tuple[float, ...]]) -> Tuple[List[float], List[float]]:
        r = [item for sublist in data[1:] for item in sublist]
        x_min, x_max = min(data[0]), max(data[0])
        y_min, y_max = min(r), max(r)

        x_range = SingleAxesPointsGroup.__adjust_range(x_min, x_max, 5)
        y_range = SingleAxesPointsGroup.__adjust_range(y_min, y_max, 10)
        return x_range, y_range

    @staticmethod
    def __adjust_range(t_min: float, t_max: float, num_grid_lines) -> List[float]:
        t_digits = extract_num_digits(t_max)
        factor = 0.1 if t_digits <= 1 else 1.0
        n_max = next_multiple(t_max - t_min, num_grid_lines)
        n_min = int(t_min)
        return  [n_min*factor, n_max*1.01*factor, factor*(n_max - n_min)/num_grid_lines]


class SingleAxesPointsScene(Scene):

    def construct(self) -> None:
        import random
        data_points = [(n, math.exp(-0.1*n) + 0.1*random.random(), math.sin(0.03*n) + 0.15*random.random())
                       for n in range(30)]
        legend_group = LegendGroup(legend_labels=[MathTex(r"Set 1", font_size=22),
                                                  MathTex(r"Set 2", font_size=22)])
        single_axes_points_group = SingleAxesPointsGroup(x_label="x",
                                                         y_label="y",
                                                         legend_group=legend_group,
                                                         title=MathTex(r" \text{Single axes plot}",
                                                                       font_size=44).to_edge(UP),
                                                         points=data_points)
        legend_group.next_to(single_axes_points_group, DOWN, buff=0.2)
        title, legend, labels, axes, *plts = single_axes_points_group.get_attributes()
        self.play(title, legend, labels, axes, plts, run_time=2)


if __name__ == '__main__':
    scene = SingleAxesPointsScene()
    scene.construct()



