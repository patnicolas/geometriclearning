__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

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

from typing import List, AnyStr
from abc import abstractmethod
import matplotlib.image as mpimg
__all__ = ['BaseAnimation']


class BaseAnimation(object):
    def __init__(self, logo_pos: List[float], interval: int, fps: int) -> None:
        """
        Constructor for the base animation
        @param logo_pos: Position of the chart used in call to plt.set_position or ax.set_position
        @type logo_pos: 4-dimension array
        @param interval: Interval in milliseconds between frames
        @type interval: int
        @param fps: Number of frame per seconds for animation
        @type fps: int
        """
        assert len(logo_pos) == 4, f'Length of chart position {len(logo_pos)} should be 4'
        assert 1 < interval <= 4096, f'Interval for animation {interval} should be [2, 4096]'
        assert 0 < fps <= 2048, f'Frame per second for animation {fps} should be [1, 2048]'

        self.chart_pos = logo_pos
        self.interval = interval
        self.fps = fps

    @abstractmethod
    def _group_name(self) -> AnyStr:
        raise NotImplementedError('_group_name has to be implemented in subclasses')

    @abstractmethod
    def draw(self, mp4_filename: AnyStr = None) -> None:
        """
            Draw and animate Lie group/manifold in ambient Euclidean space. The animation is driven by Matplotlib
            FuncAnimation class that require an update nested function.
            This method needs to be overwritten in sub-classes

            @param mp4_filename: Name of the mp4 file is to be generated (False plot are displayed but not saved)
            @type mp4_filename: str
        """
        raise NotImplementedError('draw has to be implemented in subclasses')

    def _draw_logo(self, fig) -> None:
        """
        Draw Logo on the top of the animation frame
        @param fig: Matplotlib figure
        @type fig: Figure
        """
        img = mpimg.imread('../../input/Animation_logo.png')
        inset_ax = fig.add_axes(self.chart_pos)
        inset_ax.imshow(img, alpha=1.0)
        inset_ax.axis('off')

