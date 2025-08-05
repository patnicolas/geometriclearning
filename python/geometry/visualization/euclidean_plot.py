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
from typing import List, Optional
# 3rd Party imports
import numpy as np
# Library imports
from geometry.visualization.manifold_plot import ManifoldPlot
__all__ = ['EuclideanPlot']



class EuclideanPlot(ManifoldPlot):

    def __init__(self, manifold_points: np.array, mean: Optional[np.array] = None) -> None:
        """
        Constructor for plotting Euclidean points
        @param manifold_points: List of points on the hypersphere implemented as Numpy array
        @type manifold_points: List
        """
        manifold_pts = manifold_points if mean is None else  np.vstack([manifold_points, mean])
        super(EuclideanPlot, self).__init__(manifold_pts)
        self.mean = mean

    def show(self, extra_components: Optional[np.array] = None) -> None:
        """
        Display the list of point on a data manifold on a 3D plot
        @param extra_components: Optional components to be added to the plot
        @type extra_components: Numpy array
        """
        import matplotlib.pyplot as plt

        x = [manifold_pt[0] for manifold_pt in self.manifold_points]
        y = [manifold_pt[1] for manifold_pt in self.manifold_points]
        z = [manifold_pt[2] for manifold_pt in self.manifold_points]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(x, y, z, c=z,  cmap='cool', s=120, marker='o', label='Euclidean points')
        fig.colorbar(sc, ax=ax, label='Z-axis Value')
        ax.scatter(self.mean[0], self.mean[1], self.mean[2], color='red', s=220, marker='o', label='Euclidean mean')

        ManifoldPlot._create_legend(title='Manifold points displayed in Euclidean space', ax=ax)
        ax.grid(True)
        ax.legend()
        plt.show()
