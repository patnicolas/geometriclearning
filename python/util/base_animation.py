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

from typing import AnyStr, Any, Dict
from abc import abstractmethod, ABC
import matplotlib.image as mpimg
__all__ = ['BaseAnimation']


class BaseAnimation(ABC):
    """
    Generic Abstract class for animation
    logo_pos: Tuple[int, int]   Position of the logo if one is defined
    logo_size: Tuple[int, int]  Size of the logo if one is defined
    interval: int  Interval for FuncAnimation in msec
    fps: int  Frame per second
    """
    def __init__(self, **kwargs: Dict[AnyStr, Any]) -> None:
        """
        Constructor for the base animation
        @param **kwargs: Dictionary of configuration parameters for any given animation
        @type **kwargs: Dictionary
        """
        assert 1 < kwargs.get('interval', 128) <= 4096, \
            f'Interval for animation { kwargs.get("interval", 128)} should be [2, 4096]'
        assert 0 < kwargs.get('fps', 128) <= 2048, \
            f'Frame per second for animation {kwargs.get("fps", 128)} should be [1, 2048]'
        _dict = kwargs
        self.config = _dict['kwargs']

    @abstractmethod
    def _group_name(self) -> AnyStr:
        pass

    @abstractmethod
    def draw(self, mp4_filename: AnyStr = None) -> None:
        """
        Draw and animate lie group/manifold in ambient Euclidean space. The animation is driven by Matplotlib
        FuncAnimation class that require an update nested function.
        This method needs to be overwritten in sub-classes

        @param mp4_filename: Name of the mp4 file is to be generated (False plot are displayed but not saved)
        @type mp4_filename: str
        """
        pass

    def _draw_logo(self, fig) -> None:
        """
        Draw Logo on the top of the animation frame
        @param fig: Matplotlib figure
        @type fig: Figure
        """
        img = mpimg.imread('../../input/Animation_logo.png')
        chart_pos = (self.config['logo_pos'][0],
                     self.config['logo_pos'][1],
                     self.config['logo_size'][0],
                     self.config['logo_size'][1])
        inset_ax = fig.add_axes(chart_pos)
        inset_ax.imshow(img, alpha=1.0)
        inset_ax.axis('off')

