__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import List
from abc import abstractmethod
import matplotlib.image as mpimg

class BaseAnimation(object):
    def __init__(self, chart_pos: List[float], interval: int, fps: int) -> None:
        """
        Constructor for the base animation
        @param chart_pos: Position of the chart used in call to plt.set_position or ax.set_position
        @type chart_pos: 4-dimension array
        @param interval: Interval in milliseconds between frames
        @type interval: int
        @param fps: Number of frame per seconds for animation
        @type fps: int
        """
        assert len(chart_pos) == 4, f'Length of chart position {len(chart_pos)} should be 4'
        self.chart_pos = chart_pos
        self.interval = interval
        self.fps = fps

    @abstractmethod
    def draw(self, mp4_file: bool = False) -> None:
        raise NotImplementedError('draw has to be implemented in subclasses')

    def _draw_logo(self, fig) -> None:
        """
        Draw Logo on the top of the animation frame
        @param fig: Matplotlib figure
        @type fig: Figure
        """
        img = mpimg.imread('../input/Animation_logo.png')
        inset_ax = fig.add_axes([0.01, 0.73, 0.36, 0.36])
        inset_ax.imshow(img, alpha=1.0)
        inset_ax.axis('off')

