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

from typing import AnyStr, Optional, Tuple, Self, Any, Dict
from dataclasses import dataclass, asdict
import json
import matplotlib.pyplot as plt

from plots.renderer import Renderer


@dataclass(slots=True, frozen=True)
class PlotFontDict(Renderer):
    """
    @param font_size: Size of the font
    @type font_size: int
    @param font_weight: Weight of the font (regular, bold, italic ...)
    @type font_weight: str
    @param font_color: Color of the text (default black)
    @type font_color: str
    @param font_family: Type of the font (default sans-serif)
    @type font_family: str
    """
    font_size: int
    font_weight: AnyStr
    font_color: AnyStr
    font_family: AnyStr

    def to_dict(self) -> Dict[AnyStr, Any]:
        return {'family': self.font_family, 'size': self.font_size, 'weight': self.font_weight,
                'color': self.font_color}

    def __call__(self, arg: Any) -> None:
        pass

    def to_json(self) -> AnyStr:
        """
        Convert this instance into a JSON string. Raise a ValueError in case the dictionary is malformed.

        @return: JSON representation (serialized) of this instance
        @rtype: str
        """
        return json.dumps(self.to_dict())

@dataclass(slots=True, frozen=True)
class LegendRenderer(PlotFontDict):
    def __call__(self, arg: Any) -> None:
        plt.legend(prop={'family': self.font_family,
                         'size': self.font_size,
                         'weight': self.font_weight}
                   )


@dataclass(slots=True, frozen=True)
class TextRenderer(PlotFontDict):
    """
    Configuration of the display and content of text for any given plot
    @param text: Content for text
    @type text: str
    @param font_size: Size of the font
    @type font_size: int
    @param font_weight: Weight of the font (regular, bold, italic ...)
    @type font_weight: str
    @param font_color: Color of the text (default black)
    @type font_color: str
    @param font_family: Type of the font (default sans-serif)
    @type font_family: str
    """
    text: AnyStr

    @classmethod
    def build(cls, json_config_str: AnyStr) -> Self:
        """
        Alternative constructor taking a JSON string as input to deserialize
        Raise a JSONDecodeError exception if case the JSON string is malformed

        @param json_config_str: Input JSON string
        @type json_config_str: str
        @return: Instance of the plotting text configuration
        @rtype: TextRenderer
        """
        config_dict = json.loads(json_config_str)
        return TextRenderer(**config_dict)

    def to_dict(self) -> Dict[AnyStr, Any]:
        attributes = PlotFontDict.to_dict(self)
        attributes['text'] = self.text
        return attributes

    def __call__(self, arg: Any = None) -> None:
        text, font_dict = self.text, PlotFontDict.to_dict(self)
        match arg:
            case 'title':
                plt.title(label=text, fontdict=font_dict)
            case 'xlabel':
                plt.xlabel(label=text, fontdict=font_dict)
            case 'ylabel':
                plt.ylabel(label=text, fontdict=font_dict)
            case _:
                raise NotImplementedError(f'{any} component is not supported')

    def __str__(self) -> AnyStr:
        return f'{self.text} - Font: {self.font_family}, {self.font_size}, {self.font_color}, {self.font_weight}'


@dataclass(slots=True, frozen=True)
class CommentRenderer(TextRenderer):
    """
    @param position: Position for the text if defined (e.g., comments)
    @type position : Tuple[float, float]
    """
    position: Tuple[float, float] = None

    def to_dict(self) -> Dict[AnyStr, Any]:
        attributes = super().to_dict()
        attributes['position'] = position
        return attributes

    def __str__(self) -> AnyStr:
        return f'{super.__str__(self)} - Position: {self.position}'

    def __call__(self, arg: Any = None) -> None:
        x, y = self.position
        plt.text(x=x,
                 y=y,
                 s=self.text,
                 c=self.font_color,
                 fontsize=self.font_size,
                 fontweight=self.font_weight,
                 transform=plt.gca().transAxes)


@dataclass(slots=True, frozen=True)
class PlotContext(Renderer):
    """
    @param background_color: Background color
    @type background_color: str
    @param filename: Name of the file to save the plot. The plot is automatically saved in defined,
    @type filename: str
    @param grid: Boolean flag to specify if a grid has to be drawn
    @type grid: bool
    @param fig_size: Size of the plot
    @type fig_size: Tuple[int, int]
    """
    grid: bool = True
    background_color: AnyStr = 'white'
    fig_size: Optional[Tuple[int, int]] = None
    filename: AnyStr = None

    def to_dict(self) -> Dict[AnyStr, Any]:
        return {'grid': self.grid,
                'background_color': self.background_color,
                'fig_size': self.fig_size,
                'filename': self.filename}

    def __call__(self, arg: Any = None) -> None:
        fig = plt.figure(figsize=self.fig_size)
        fig.set_facecolor(self.background_color)
        plt.grid(self.grid)
        if self.filename is not None:
            fig.savefig(f'{self.filename}.png')


@dataclass(slots=True, frozen=True)
class AnnotationRenderer(Renderer):
    text: AnyStr
    xy: Tuple[int, int]
    xytext: Tuple[int, int]
    color: AnyStr
    arrow_style: AnyStr
    connection_style: AnyStr

    def to_dict(self) -> Dict[AnyStr, Any]:
        return asdict(self)

    def __call__(self, arg: Any = None) -> None:
        plt.annotate(text=self.text,
                     xy=self.xy,
                     xytext=self.xytext,
                     color=self.color,
                     arrowprops=dict(arrowstyle=self.arrow_style,
                                     color='black',
                                     connectionstyle=self.connection_style))


class PlottingDefaults(object):
    markers = ['-', '--', '-.', '--', ':', '-']
    colors = ['blue', 'green', 'red', 'black', 'grey', 'orange']


@dataclass(slots=True, frozen=True)
class PlotsRenderer(Renderer):
    data_dict: Dict[AnyStr, Any]

    def to_dict(self) -> Dict[AnyStr, Any]:
        return self.data_dict

    def __call__(self, arg: Any = None) -> None:
        iterator = iter(self.data_dict.items())
        _, x_values = next(iterator)

        # Draws the multiple plots
        count = 0
        for y_label, y_value in iterator:
            plt.plot(x_values,
                     y_value,
                     label=y_label,
                     color=PlottingDefaults.colors[count],
                     linestyle=PlottingDefaults.markers[count])
            count += 1

