__author__ = "Patrick R. Nicolas"
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

# Standard Library imports
from typing import AnyStr
from abc import ABC, abstractmethod
# 3rd Party imports
from matplotlib.axes import Axes
import persim


class PersistenceDiagramType(ABC):
    def __init__(self, ax: Axes) -> None:
        self.ax = ax

    @abstractmethod
    def display(self, title: AnyStr, x_label: AnyStr, y_label: AnyStr) -> None:
        pass

    def _set_plot_env(self, title: AnyStr, x_label: AnyStr, y_label: AnyStr) -> None:
        self.ax.set_xlabel(x_label, fontdict={'family': 'serif', 'size': 11, 'style': 'italic'})
        self.ax.set_ylabel(y_label, fontdict={'family': 'serif', 'size': 11, 'style': 'italic'})
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_title(label=title, fontdict={'family': 'serif', 'size': 14, 'color': 'blue'})


class BirthDeathDiagram(PersistenceDiagramType):
    def __init__(self, ax: Axes, diagram_data) -> None:
        super(BirthDeathDiagram, self).__init__(ax)
        self.diagram_data = diagram_data

    def display(self, title: AnyStr, x_label: AnyStr, y_label: AnyStr) -> None:
        self._set_plot_env(title=title, x_label=x_label, y_label=y_label)
        persim.plot_diagrams(show=True, diagrams=self.diagram_data, ax=self.ax)


class PersistenceImage(PersistenceDiagramType):
    def __init__(self, ax: Axes, diagram_data) -> None:
        super(PersistenceImage, self).__init__(ax)
        self.diagram_data = diagram_data

    def display(self, title: AnyStr, x_label: AnyStr, y_label: AnyStr) -> None:
        import ripser
        from persim import PersistenceImager

        dgms = ripser.ripser(self.diagram_data)['dgms']
        pimager = PersistenceImager(pixel_size=0.3)
        input_data = dgms[1:3]
        pimager.fit(input_data)
        output_image = pimager.transform(input_data)
        self.ax.matshow(output_image[0].T, **{"origin": "lower"})
        self._set_plot_env(title=title, x_label=x_label, y_label=y_label)


