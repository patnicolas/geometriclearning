__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from manim import *
import numpy as np



class ConvVGroup(VGroup):
    def __init__(self, shift: float, scale: float, opacity: float, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)

        grid_lines = ConvVGroup.__add_grid(self)
        self.add(grid_lines)

        receptive_fields = ConvVGroup.__add_receptive_fields(self, grid_lines)
        self.add(receptive_fields)
        self.shift(LEFT*shift)
        self.scale(scale)
        self.set_opacity(opacity)

    @staticmethod
    def __add_receptive_fields(conf_vgroup: VGroup, grid: VGroup) -> VGroup:
        fields = VGroup()
        start_cube = Cube(side_length=0.24, color=RED).shift(LEFT).shift(UP)
        colors = ['BLUE', 'RED', "WHITE", "GREEN"]
        for n in range(-2, 2):
            prev_cube = start_cube
            for m in range(-2, 2):
                this_color = colors[(m+n+4) % 4]
                cube = Cube(side_length=0.24, color=this_color, fill_color=this_color, fill_opacity=0.2)
                cube.next_to(prev_cube, RIGHT, buff=0.03)
                prev_cube = cube
                fields.add(cube)
            start_cube.next_to(start_cube, DOWN, buff=0.03)

        fields = fields.next_to(grid, OUT, buff=0.3)
        fields.z_index = len(conf_vgroup) + 1
        return fields


    @staticmethod
    def __add_grid(conf_vgroup: VGroup) -> VGroup:
        grid_lines = VGroup()
        for y in range(-8, 9):
            grid_lines.add(Line3D(start=np.array([-8, y, 0]),
                                  end=np.array([8, y, 0]),
                                  thickness=0.04,
                                  color=GRAY))
        for x in range(-8, 9):
            grid_lines.add(Line3D(start=np.array([x, -8, 0]),
                                  end=np.array([x, 8, 0]),
                                  thickness=0.04,
                                  color=GRAY))
        grid_lines = grid_lines.scale(0.35)
        grid_lines.z_index = len(conf_vgroup) + 1
        return grid_lines
