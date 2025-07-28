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

from abc import ABC, abstractmethod
from typing import List, AnyStr, Optional

from matplotlib.collections import PathCollection
from mpl_toolkits.mplot3d import Axes3D
from geometry import GeometricException
import numpy as np


class ManifoldPlot(ABC):
    def __init__(self,  manifold_points: np.array) -> None:
        """
        Constructor for plotting points on manifold
        @param manifold_points: List of points on the hypersphere implemented as torch Tensors
        @type manifold_points: List of Numpy arrays representing points on the manifold
        """
        if len(manifold_points) < 1:
            raise GeometricException('Cannot display undefined number of data points')
        self.manifold_points = manifold_points

    @abstractmethod
    def show(self, extra: Optional[np.array] = None) -> None:
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
