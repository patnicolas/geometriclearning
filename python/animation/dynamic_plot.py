__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from manim import *
from typing import Callable, AnyStr



class LossFunctionPlot(object):
    def __init__(self,
                 f: Callable[[float], float],
                 func_label: AnyStr,
                 max_x: int,
                 max_y: int,
                 line_color=LIGHT_GREY) -> None:
        self.loss = f
        self.max_x = max_x
        self.max_y = max_y
        self.func_label = func_label
        self.line_color = line_color

    def get_plane(self) -> NumberPlane:
        background_line_style = {
            "stroke_color": self.line_color,
            "stroke_width": 1,
            "stroke_opacity": 0.9
        }
        return NumberPlane(x_range=[0, self.max_x],
                           y_range=[0, self.max_y],
                           y_length=5,
                           x_length=5,
                           background_line_style=background_line_style).add_coordinates().shift(RIGHT*4)

    def get_graph(self) -> (ParametricFunction, MathTex):
        plane = self.get_plane()
        graph = plane.plot(function=self.loss, x_range=[0, self.max_x], color=YELLOW, colorscale_axis=2)
        graph_lab = MathTex(self.func_label).next_to(graph, UP, buff=0.15).set_color(WHITE).scale(0.8)
        return plane, graph, graph_lab



class DynamicPlotScene(Scene):
    def construct(self):
        def f(x: float) -> float:
            import random
            return 3.0 / (x + 1) * (1.0 + 0.4 * random.uniform(a=0.0, b=1.0))
        loss_function = LossFunctionPlot(f=f, func_label=r'Loss\ Function', max_x=20, max_y=4)
        plane, graph, graph_lab = loss_function.get_graph()

        self.play(
            DrawBorderThenFill(plane),
            run_time=1,
            lag_ratio=0.1)
        self.add(graph_lab)
        self.play(Create(graph_lab))
        self.play(
            Create(graph),
            run_time=8,
            lag_ratio=0.3)
        self.wait()


if __name__ == '__main__':
    scene = DynamicPlotScene()
    scene.construct()
