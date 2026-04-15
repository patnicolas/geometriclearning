from manim import *
from typing import Callable, Tuple
from . import Rnge
import numpy as np

class ParametricCurveGroup(VGroup):
    def __init__(self,
                 func: Callable[[float], np.array],
                 t_range: Rnge,
                 scale: float,
                 title: MathTex,
                 **kwargs) -> None:
        super(ParametricCurveGroup, self).__init__(**kwargs)

        self.param_func = ParametricFunction(func, t_range, color=BLUE).scale(scale)
        self.title = title
        self.add(title, self.param_func)

    def get_attributes(self) -> Tuple[Write, Create]:
        return Write(self.title, run_time=1), Create(self.param_func, run_time=4)

