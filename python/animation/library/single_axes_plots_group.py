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
from typing import Callable, Tuple, List, AnyStr, Any
import numpy as np
from animation.library import colors, get_num_ticks
from legend_group import LegendGroup


class SingleAxesPlotsGroup(VGroup):

    def __init__(self,
                 x_range: List[float],
                 y_range: List[float],
                 x_label: AnyStr,
                 y_label: AnyStr,
                 title: MathTex,
                 legend_group: LegendGroup,
                 funcs: Tuple[Callable[[float], np.array]],
                 **kwargs) -> None:
        super(SingleAxesPlotsGroup, self).__init__(**kwargs)

        self.ax = NumberPlane(x_range=x_range,
                              y_range=y_range,
                              x_length=8,
                              y_length=4,
                              background_line_style={
                                  "stroke_color": LIGHT_GREY,
                                  "stroke_width": 4,
                                  "stroke_opacity": 0.6
                              },
                              axis_config={
                                 "font_size": 29,
                                 "include_numbers": True,  # This is often more reliable than .add_coordinates()
                                 "decimal_number_config": {"color": LIGHT_GREY}  # Force a bright color
                       }
                    )
        self.labels = self.ax.get_axis_labels(x_label=x_label, y_label=y_label)
        self.curves = [self.ax.plot(func, color=colors[idx]) for idx, func in enumerate(funcs)]
        self.title = title
        self.legends_group = legend_group
        self.add(title, self.legends_group, self.ax, self.labels, self.curves)

    def get_attributes(self) -> Tuple[Write | Create, ...]:
        return ((Write(self.title), Write(self.legends_group), Write(self.labels), Create(self.ax))
                + tuple([Create(curve) for curve in self.curves]))


class SingleAxesPlotsScene(Scene):
    def construct(self) -> None:
        funcs = [lambda x: np.exp(-x), lambda x: np.exp(-0.1 * x), lambda x: np.exp(-0.5 * x)]
        legend_group = LegendGroup(legend_labels=[MathTex(r"exp(-x)", font_size=28),
                                                  MathTex(r"exp(-0.1x)", font_size=28),
                                                  MathTex(r"exp(-0.5x)", font_size=28)],
                                   corner=UR+1,
                                   shift=LEFT*0.4)
        single_axes_plots_group = SingleAxesPlotsGroup(x_range=[0, 8, 1],
                                                       y_range=[0, 1, 0.5],
                                                       x_label="x",
                                                       y_label="y",
                                                       legend_group=legend_group,
                                                       title=MathTex(r" \text{Single axes plot}",
                                                                     font_size=44).to_edge(UP),
                                                       funcs=funcs)
        box = SurroundingRectangle(single_axes_plots_group,
                                   color=DARK_GREY,
                                   buff=-0.8,
                                   fill_opacity=0.2,
                                   fill_color=BLACK)
        title, legend, labels, axes, *plts = single_axes_plots_group.get_attributes()
        # self.add(single_axes_multi_plots)
        self.add(box)
        single_axes_plots_group.scale(scale_factor=0.7)
        self.play(title, legend, labels, axes, plts, run_time=3)


if __name__ == '__main__':
    single_axes_multi_plots_scene = SingleAxesPlotsScene()
    single_axes_multi_plots_scene.construct()

