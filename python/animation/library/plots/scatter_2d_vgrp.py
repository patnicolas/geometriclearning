
from manim import *
from typing import List, Tuple, AnyStr, Callable, Any
from animation.library.plots import get_2d_ranges
from animation.library.plots.legend_vgrp import LegendVGrp, LegendType
from dataclasses import dataclass

@dataclass
class Scatter2DConfig:
    xy_labels: Tuple[AnyStr, AnyStr]
    lengths: Tuple[int, int]
    title: MathTex
    num_lines: int
    radius: float
    label_font_size: int
    axis_font_size: int
    legend_texts: Tuple[AnyStr, ...]

    def get_legend_group(self) -> LegendVGrp:
        legend_labels = [MathTex(rf"{legend}", font_size=22) for legend in self.legend_texts]
        return LegendVGrp(legend_labels=legend_labels,
                          legend_type=LegendType.DOT,
                          radius=self.radius,
                          arrange=LEFT,
                          buff=0.4)


class Scatter2DVGrp(VGroup):

    def __init__(self,
                 scatter_2d_config: Scatter2DConfig,
                 data_points: List[Tuple[float, ...]],
                 **kwargs) -> None:
        super(Scatter2DVGrp, self).__init__(**kwargs)

        xy_ranges = get_2d_ranges(list(zip(*data_points)), scatter_2d_config.num_lines)
        self.data_points = data_points

        self.ax = NumberPlane(x_range=xy_ranges[0],
                              y_range=xy_ranges[1],
                              x_length=scatter_2d_config.lengths[0],
                              y_length=scatter_2d_config.lengths[1],
                              background_line_style={
                                  "stroke_color": DARK_GREY,
                                  "stroke_width": 3,
                                  "stroke_opacity": 0.6
                              },
                              x_axis_config={"label_direction": DOWN, "line_to_number_buff": 0.15},
                              y_axis_config={"label_direction": LEFT, "line_to_number_buff": 0.15},
                              axis_config={
                                  "font_size": scatter_2d_config.axis_font_size,
                                  "include_ticks": True,
                                  "include_numbers": True,
                                  "decimal_number_config": {"color": LIGHT_GREY},
                              })

        self.dots = VGroup()
        self.labels = self.ax.get_axis_labels(x_label=MathTex(scatter_2d_config.xy_labels[0],
                                                              font_size=scatter_2d_config.label_font_size),
                                              y_label=MathTex(scatter_2d_config.xy_labels[1],
                                                              font_size=scatter_2d_config.label_font_size))
        self.title = scatter_2d_config.title.next_to(self.ax, UP, buff=0.4)

        self.legends_group = scatter_2d_config.get_legend_group()
        self.legends_group.next_to(self.title, RIGHT+DOWN, buff=0.2)
        self.add(self.ax, self.labels, self.title, self.legends_group, self.dots)

    def get_updater(self, vt) -> Callable[[Any], Any]:
        x_data, y_data = zip(*self.data_points)

        def updater(obj):
            idx = int(vt.get_value())
            obj.dots.become(
                VGroup(*[
                    Dot(obj.ax.c2p(x_data[i], y_data[i]))
                    for i in range(idx)
                ])
            )
        return updater

    def get_dynamic(self) -> Create:
        return Create(self.dots)


class Scatter2DDynamicScene(Scene):
    data_points: List[Tuple[float, float]] = [(1.0, 14.0), (2.0, 6.0), (3.2, 1.7), (4.0, 3.5), (1.1, 2.9), (0.0, 4.0),
                                              (3.7, 9.0), (4.4, 3.6)]
    run_time = 4

    def construct(self) -> None:
        vt = ValueTracker(0)

        legend_group, scatter_2d_config = Scatter2DDynamicScene.get_config()
        scatter_2d_group = Scatter2DVGrp(scatter_2d_config=scatter_2d_config,
                                         data_points=Scatter2DDynamicScene.data_points).to_edge(LEFT)
        legend_group.next_to(scatter_2d_group, DOWN, buff=0.2)
        scatter_2d_group.add_updater(scatter_2d_group.get_updater(vt))
        self.add(scatter_2d_group)

        self.play(vt.animate.set_value(len(Scatter2DDynamicScene.data_points)-1),
                  run_time=Scatter2DDynamicScene.run_time,
                  rate_func=linear)
        self.wait()




if __name__ == '__main__':
    scene = Scatter2DDynamicScene()
    scene.construct()
