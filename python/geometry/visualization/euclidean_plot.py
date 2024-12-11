__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from geometry.visualization.manifold_plot import ManifoldPlot
from typing import List
import numpy as np
from geometry import GeometricException


class EuclideanPlot(ManifoldPlot):

    def __init__(self, manifold_points: List[np.array]) -> None:
        """
        Constructor for plotting Euclidean points
        @param manifold_points: List of points on the hypersphere implemented as Numpy array
        @type manifold_points: List
        """
        super(EuclideanPlot, self).__init__(manifold_points)

    def show(self) -> None:
        """

        """
        if len(self.manifold_points) < 1:
            raise GeometricException('Cannot display undefined number of data points')

        import matplotlib.pyplot as plt

        x = [manifold_pt[0] for manifold_pt in self.manifold_points]
        y = [manifold_pt[1] for manifold_pt in self.manifold_points]
        z = [manifold_pt[2] for manifold_pt in self.manifold_points]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(x, y, z, c=z,  cmap='cool', s=120, marker='o', label='Euclidean points')
        fig.colorbar(sc, ax=ax, label='Z-axis Value')

        ManifoldPlot._create_legend('Manifold points displayed in Euclidean space', ax)
        ax.grid(True)
        ax.legend()
        plt.show()
