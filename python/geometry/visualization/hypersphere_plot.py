__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from geometry.visualization.manifold_plot import ManifoldPlot
from typing import List
import numpy as np


class HyperspherePlot(ManifoldPlot):

    def __init__(self, manifold_points: List[np.array]) -> None:
        """
        Constructor for plotting Hypersphere points
        @param manifold_points: List of points on the hypersphere implemented as torch Tensors
        @type manifold_points: List of Numpy arrays representing points on the manifold
        @type manifold_points: List
        """
        super(HyperspherePlot, self).__init__(manifold_points)

    def show(self) -> None:
        """
        Plot the manifold points on a 3D plot
        """
        import matplotlib.pyplot as plt
        import geomstats.visualization as visualization

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Walk through the list of data point on the manifold
        for idx, manifold_pt in enumerate(self.manifold_points):
            ax = visualization.plot(
                manifold_pt,
                ax=ax,
                space="S2",
                s=160,
                alpha=0.8,
                label=f'pt_{idx}')
        ManifoldPlot._create_legend(title ='Manifold points displayed on S2', ax=ax)
        plt.show()
