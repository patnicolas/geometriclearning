__author__ = "Patrick R. Nicolas"
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

# Standard Library imports
from typing import AnyStr
from abc import ABC, abstractmethod
# 3rd Party imports
from matplotlib.axes import Axes
import persim
import numpy as np


class PersistenceDiagramType(ABC):
    """
    Abstract base class to generate any kind of persistence diagrams as supported by the scikit-learn TDA library
    """
    __slots__ = ['ax']

    def __init__(self, ax: Axes) -> None:
        """
        Constructor for the abstract base class to any current or future Persistence diagrams
        @param ax: Matplotlib axes
        @type ax: Axes
        """
        self.ax = ax

    @abstractmethod
    def display(self, title: AnyStr, x_label: AnyStr, y_label: AnyStr) -> None:
        """
        Display the persistence diagram given a title and labels for X and Y axes
        @param title: Title or description of the persistence diagram plot
        @type title: AnyStr
        @param x_label: Label for X-axis
        @type x_label: AnyStr
        @param y_label: Label of Y-axes
        @type y_label: AnyStr
        """
        pass

    def _set_plot_env(self, title: AnyStr, x_label: AnyStr, y_label: AnyStr) -> None:
        self.ax.set_xlabel(x_label, fontdict={'family': 'serif', 'size': 11, 'style': 'italic'})
        self.ax.set_ylabel(y_label, fontdict={'family': 'serif', 'size': 11, 'style': 'italic'})
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_title(label=title, fontdict={'family': 'serif', 'size': 14, 'color': 'blue'})


class BirthDeathDiagram(PersistenceDiagramType):
    """
    A persistence diagram maps the “birth” (appearance) and “death” (disappearance) times/scales of topological 
    features such as cavities, holes or connected components in a dataset as a collection of points in a 2D plot. 
    """
    def __init__(self, ax: Axes, diagram_data: np.array) -> None:
        """
        Constructor for the persistence diagram (birth & death of simplicial complexes)
        @param ax: Plot axes
        @type ax: Axes
        @param diagram_data: Data to be diplay in this diagram
        @type diagram_data: np.array
        """
        super(BirthDeathDiagram, self).__init__(ax)
        self.diagram_data = diagram_data

    def display(self, title: AnyStr, x_label: AnyStr, y_label: AnyStr) -> None:
        """
        Display the persistence diagram - Birth and Death of simplices given a title and labels for X and Y axes
        @param title: Title or description of the persistence diagram plot
        @type title: AnyStr
        @param x_label: Label for X-axis
        @type x_label: AnyStr
        @param y_label: Label of Y-axes
        @type y_label: AnyStr
        """
        self._set_plot_env(title=title, x_label=x_label, y_label=y_label)
        persim.plot_diagrams(show=True, diagrams=self.diagram_data, ax=self.ax)


class PersistenceImage(PersistenceDiagramType):
    """
    Class that wraps the generation of Persistence images. Persistence images transform persistence diagrams into
    gridded numerical images that can used as feature vectors to classical machine learning algorithms or deep learning
    architecture.
    Bright colors represent high persistence while fainted colors represents low persistence. Persistence images
    preserve essential topological structure while giving a stable, computable representation.
    """
    def __init__(self, ax: Axes, diagram_data: np.array) -> None:
        """
        Constructor for the persistence image
        @param ax: Plot axes
        @type ax: Axes
        @param diagram_data: Data to be diplay in this diagram
        @type diagram_data: np.array
        """
        super(PersistenceImage, self).__init__(ax)
        self.diagram_data = diagram_data

    def display(self, title: AnyStr, x_label: AnyStr, y_label: AnyStr) -> None:
        """
        Display the persistence image given a title and labels for X and Y axes. This implementation relies
        on ripser and persim libraries

        @param title: Title or description of the persistent image
        @type title: AnyStr
        @param x_label: Label for X-axis
        @type x_label: AnyStr
        @param y_label: Label of Y-axes
        @type y_label: AnyStr
        """
        import ripser
        from persim import PersistenceImager

        dgms = ripser.ripser(self.diagram_data)['dgms']
        pimager = PersistenceImager(pixel_size=0.3)
        input_data = dgms[1:3]
        pimager.fit(input_data)
        output_image = pimager.transform(input_data)
        self.ax.matshow(output_image[0].T, **{"origin": "lower"})
        self._set_plot_env(title=title, x_label=x_label, y_label=y_label)


class PersistenceLandscape(PersistenceDiagramType):
    """
    Class that wraps persistence landscape
    Persistence landscapes turn persistence diagrams into functions in a Hilbert space that preserve topological
    information. It is essentially a summary of how many features are alive at each scale and how strong they are.
    Usually the tail values highly the important topological features.
    The degree of homology groups are
        0 for connected points
        1 for loops
        2 for cavities
    """
    def __init__(self, ax: Axes, diagram_data: np.array, hom_degree: int, depth: int, is_exact: bool) -> None:
        """
        Constructor for the Approximation and Exact Persistence Landscape plots
        @param ax: Plot axes
        @type ax: Axes
        @param diagram_data: Data input to the persistence landscape
        @type diagram_data: Numpy array
        @param hom_degree: Number of degree of the homology group to be represented.
        @type hom_degree: int
        @param depth: Number of landscape curves (lambdas) ordered by increasing values 
        @type depth: int
        @param is_exact: It is an exact persistence landscape or an approximation landscape
        @type is_exact: bool
        """
        import ripser

        super(PersistenceLandscape, self).__init__(ax)
        self.diagram_data = ripser.ripser(diagram_data)['dgms']
        self.hom_degree = hom_degree
        self.depth = depth
        self.is_exact = is_exact

    def display(self, title: AnyStr, x_label: AnyStr, y_label: AnyStr) -> None:
        """
            Display the persistent landscape given a title and labels for X and Y axes. This implementation relies
            on the persim library

            @param title: Title or description of the persistent exact or approximate landscapce
            @type title: AnyStr
            @param x_label: Label for X-axis
            @type x_label: AnyStr
            @param y_label: Label of Y-axes
            @type y_label: AnyStr
            """
        from persim import PersLandscapeApprox, PersLandscapeExact
        from persim.landscapes import plot_landscape_simple

        self._set_plot_env(title=title, x_label=f'{x_label} {self.hom_degree}', y_label=y_label)
        pla = PersLandscapeExact(dgms=self.diagram_data, hom_deg=self.hom_degree) if self.is_exact \
            else PersLandscapeApprox(dgms=self.diagram_data, hom_deg=self.hom_degree)
        plot_landscape_simple(pla, ax=self.ax, depth_range=range(self.depth))



