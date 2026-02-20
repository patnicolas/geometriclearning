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

@dataclass(slots=True, frozen=True)
class PlotFontDict:
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
        return {'family': self.font_family, 'size': self.font_size, 'weight': self.font_weight, 'color': self.font_color}

    def to_json(self) -> AnyStr:
        """
        Convert this instance into a JSON string. Raise a ValueError in case the dictionary is malformed.

        @return: JSON representation (serialized) of this instance
        @rtype: str
        """
        return json.dumps(self.to_dict())

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

    def to_dict(self) -> (AnyStr, Dict[AnyStr, Any]):
        attributes = PlotFontDict.to_dict(self)
        attributes['text'] = self.text
        return attributes

    def __call__(self, arg: Any = None) -> None:
        text, font_dict = self.text, PlotFontDict.to_dict(self)
        match any:
            case 'title':
                plt.title(label=text, fontdict=font_dict)
            case 'xlabel':
                plt.xlabel(label=text, fontdict=font_dict)
            case 'ylabel':
                plt.ylabel(label=text, fontdict=font_dict)

    def __str__(self) -> AnyStr:
        return f'{self.text} - Font: {self.font_family}, {self.font_size}, {self.font_color}, {self.font_weight}'


@dataclass(slots=True, frozen=True)
class CommentRenderer(TextRenderer):
    """
    @param position: Position for the text if defined (e.g., comments)
    @type position : Tuple[float, float]
    """
    position: Tuple[float, float] = None

    def to_json(self) -> (AnyStr, AnyStr):
        return self.text, json.dumps(asdict(self))

    def __str__(self) -> AnyStr:
        return f'{super.__str__(self)} - Position: {self.position}'

    def draw(self, arg: Any = None) -> None:
        x, y = self.position
        plt.text(x=x,
                 y=y,
                 s=self.text,
                 c=self.font_color,
                 fontsize=self.font_size,
                 fontweight=self.font_weight,
                 transform=plt.gca().transAxes)

@dataclass(slots=True, frozen=True)
class PlotContext:
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
class AnnotationRenderer:
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
class PlotsRenderer:
    data_dict: Dict[AnyStr, Any]

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


@dataclass(slots=True, frozen=True)
class PlottingConfig:
    """
    Generic data class for configuration any plot, independently of the plotting library or engine. The plot is
    automatically saved if the variable, filename is defined.
    
    @param plot_type: Type of the plot (scatter, line plot, bar chart ...).
    @type plot_type: str
    @param title_config: Configuration for the title of the plot
    @type title_config: TextRenderer
    @param x_label_config: Configuration for the X-label of the plot
    @type x_label_config: TextRenderer
    @param y_label_config: Configuration for the y-label of the plot
    @type y_label_config: CommentRenderer
    @param comment_config: Configuration for comment the plot
    @type comment_config: CommentRenderer
    @param multi_plot_pause: Pause in millis between display of plots
    @type multi_plot_pause: float
    """
    plot_type: AnyStr
    plot_context: PlotContext
    plot_renderer: PlotsRenderer
    title_config: TextRenderer
    x_label_config: TextRenderer
    y_label_config: TextRenderer
    legend_config: PlotFontDict
    comment_config: CommentRenderer = None
    annotate_config: AnnotationRenderer = None
    multi_plot_pause: float = 0.0

    @classmethod
    def build(cls, json_config_str: AnyStr) -> Self:
        """
        Alternative constructor taking a JSON string as input to deserialize
        Raise a JSONDecodeError exception if case the JSON string is malformed

        @param json_config_str: Input JSON string
        @type json_config_str: str
        @return: Instance of the plotting tconfiguration
        @rtype: PlottingConfig
        """
        config_dict = json.loads(json_config_str)
        return PlottingConfig(**config_dict)

    def to_json(self) -> AnyStr:
        return json.dumps(asdict(self), indent=2)

    def draw(self, arg: Any) -> None:
        self.plot_context.__call__()
        self.plot_renderer()
        self.title_config()
        self.x_label_config()
        self.y_label_config()
        if self.comment_config is not None:
            self.comment_config()
        if self.annotate_config is not None:
            self.annotate_config()
        plt.show()

    def __str__(self) -> AnyStr:
        return (f'\nType: {self.plot_type}\nTitle: {self.title_config}\nX-label: {self.x_label_config}'
                f'\nY-Label: {self.y_label_config}\nLegend: {self.legend_config}\nContext: {self.plot_context}'
                f'\nCommment: {self.comment_config}')
