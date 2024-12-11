__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from abc import ABC, abstractmethod
from typing import List, AnyStr, Optional

from matplotlib.collections import PathCollection
from mpl_toolkits.mplot3d import Axes3D
from geometry import GeometricException
import numpy as np


class ManifoldPlot(ABC):
    def __init__(self,  manifold_points: List[np.array]) -> None:
        """
        Constructor for plotting points on manifold
        @param manifold_points: List of points on the hypersphere implemented as torch Tensors
        @type manifold_points: List of Numpy arrays representing points on the manifold
        """
        if len(manifold_points) < 1:
            raise GeometricException('Cannot display undefined number of data points')
        self.manifold_points = manifold_points

    @abstractmethod
    def show(self) -> None:
        pass

    @staticmethod
    def _create_legend(title: AnyStr, ax: Axes3D) -> None:
        font_dict = {'family': 'sans-serif', 'color': 'blue', 'weight': 'bold', 'size': 16}
        ax.set_title(label=title, fontdict=font_dict)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid()
        ax.legend()
