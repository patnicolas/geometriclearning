from manim import *
from animation.library.plots.parametric_curve_vgrp import ParametricCurveVGrp
from animation.library.plots.parametric_surface_vgrp import ParametricSurfaceVGrp
import numpy as np


class ParametricCurveScene(ThreeDScene):
    def construct(self):
        k = 8
        this_func = lambda t: np.array([
                np.cos(k * t) * np.cos(t),
                np.cos(k * t) * np.sin(t),
                t
            ])
        math_pattern = ParametricCurveVGrp(func=this_func,
                                           t_range=(0, TAU),
                                           scale=2.0,
                                           title=MathTex(r"r = \cos(", str(k), r"\theta)").to_edge(UP))
        title, param_curve = math_pattern.get_attributes()
        self.play(title, param_curve)
        self.wait(2)

class ParametricSurfaceScene(ThreeDScene):
    def construct(self):
        R, r = 3, 1
        this_func = lambda u, v: np.array([
            (R + r * np.cos(v)) * np.cos(u),
            (R + r * np.cos(v)) * np.sin(u),
            r * np.sin(v)
            ])
        math_pattern = ParametricSurfaceVGrp(func=this_func,
                                             u_range=(0, TAU),
                                             v_range=(0, TAU),
                                             scale=0.5,
                                             resolution=(32, 32),
                                             title=MathTex(r"r = \cos(", str(r), r"\theta)").to_edge(UP))
        title, param_surface = math_pattern.get_attributes()
        self.play(title, param_surface)
        self.wait(2)


# To run this, install manim (pip install manim) and run:
# manim -pql myscript.py GravitationalEquivariance

if __name__ == '__main__':
    # parametric_curve_example = ParametricCurveScene()
    # parametric_curve_example.construct()

    parametric_surface_scene = ParametricSurfaceScene()
    parametric_surface_scene.construct()
