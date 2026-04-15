from manim import *
from typing import Callable, Tuple
from . import Rnge
import numpy as np


class ParametricSurfaceGroup(VGroup):
    def __init__(self,
                 func: Callable[[float, float], np.array],
                 u_range: Rnge,
                 v_range: Rnge,
                 scale: float,
                 resolution: Tuple[int, int],
                 title: MathTex,
                 **kwargs) -> None:
        super(ParametricSurfaceGroup, self).__init__(**kwargs)
        self.func = func
        self.u_range = u_range
        self.v_range = v_range

        self.surface = Surface(func, u_range, v_range, resolution).scale(scale)
        self.title = title
        self.add(title, self.surface)

    def get_attributes(self) -> Tuple[Write, Create]:
        return Write(self.title, run_time=1), Create(self.surface, run_time=4)