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

from typing import AnyStr, Optional, Tuple, Self
from dataclasses import dataclass, asdict
import json


@dataclass(slots=True)
class PlottingTextConfig:
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
    @param font_type: Type of the font (default sans-serif)
    @type font_type: str
    @param position: Position for the text if defined (e.g., comments)
    @type position : Tuple[float, float]
    """
    text: AnyStr
    font_size: int
    font_weight: AnyStr
    font_color: AnyStr = 'black'
    font_type: AnyStr = 'sans-serif'
    position: Tuple[float, float] = None

    @classmethod
    def build(cls, json_config_str: AnyStr) -> Self:
        """
        Alternative constructor taking a JSON string as input to deserialize
        Raise a JSONDecodeError exception if case the JSON string is malformed
        
        @param json_config_str: Input JSON string
        @type json_config_str: str
        @return: Instance of the plotting text configuration
        @rtype: PlottingTextConfig
        """
        config_dict = json.loads(json_config_str)
        return PlottingTextConfig(**config_dict)

    def to_json(self) -> AnyStr:
        """
        Convert this instance into a JSON string. Raise a ValueError in case the dictionary is malformed.
        
        @return: JSON representation (serialized) of this instance
        @rtype: str
        """
        return json.dumps(asdict(self))

    def __call__(self) -> (AnyStr, AnyStr):
        return self.text, {'family': self.font_type, 'size': self.font_size, 'weight': self.font_weight}

    def __str__(self) -> AnyStr:
        return f'{self.text} - Font: {self.font_type}, {self.font_size}, {self.font_color}, {self.font_weight}'


@dataclass(slots=True)
class PlottingConfig:
    """
    Generic data class for configuration any plot, independently of the plotting library or engine. The plot is
    automatically saved if the variable, filename is defined.
    
    @param plot_type: Type of the plot (scatter, line plot, bar chart ...).
    @type plot_type: str
    @param background_color: Background color
    @type background_color: str
    @param title_config: Configuration for the title of the plot
    @type title_config: PlottingTextConfig
    @param x_label_config: Configuration for the X-label of the plot
    @type x_label_config: PlottingTextConfig
    @param y_label_config: Configuration for the y-label of the plot
    @type y_label_config: PlottingTextConfig
    @param comment_config: Configuration for comment the plot
    @type comment_config: PlottingTextConfig
    @param legend_font_size: Size of font for legend
    @type legend_font_size: int
    @param filename: Name of the file to save the plot. The plot is automatically saved in defined,
    @type filename: str
    @param color_palette: Color palette (default dee[)
    @type color_palette: str
    @param fig_size: Size of the plot
    @type fig_size: Tuple[int, int]
    @param multi_plot_pause: Pause in millis between display of plots
    @type multi_plot_pause: float
    """
    plot_type: AnyStr
    title_config: PlottingTextConfig
    x_label_config: PlottingTextConfig
    y_label_config: PlottingTextConfig
    comment_config: PlottingTextConfig = None
    background_color: AnyStr = 'white'
    legend_font_size: int = None
    filename: AnyStr = None
    grid: bool = True
    color_palette: AnyStr = 'deep'
    fig_size: Optional[Tuple[int, int]] = None
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

    def get_legend_font_size(self) -> int:
        return self.x_label_config.font_size if self.legend_font_size is None else self.legend_font_size

    def get_fig_size(self) -> (int, int):
        return self.fig_size if self.fig_size is not None else (10, 8)

    def __str__(self) -> AnyStr:
        return (f'\nType: {self.plot_type}\nTitle: {self.title_config}\nX-label: {self.x_label_config}'
                f'\nY-Label: {self.y_label_config}\nPalette: {self.color_palette}\nFilename: {self.filename}')
