__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from geometry.visualization.manifold_plot import ManifoldPlot
from typing import List, Optional
import numpy as np


class HyperspherePlot(ManifoldPlot):

    def __init__(self, manifold_points: List[np.array], mean: Optional[np.array] = None) -> None:
        """
        Constructor for plotting Hypersphere points
        @param manifold_points: List of points on the hypersphere implemented as torch Tensors
        @type manifold_points: List of Numpy arrays representing points on the manifold
        @type manifold_points: List
        """
        super(HyperspherePlot, self).__init__(manifold_points)
        self.mean = mean

    def show(self, extra_components: Optional[np.array] = None) -> None:
        """
        Plot the manifold points on a 3D sphere with a 3D grid as a background
        @param extra_components: Optional components to be added to the plot
        @type extra_components: Numpy array
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
                color='black',
                s=120,
                alpha=1.0,
                label=f'data {idx}')

        if extra_components is not None:
            for idx, component in enumerate(extra_components):
                ax = visualization.plot(component, space="S2", ax=ax, s=20, alpha=1.0, label=f'Component {idx}')

        # If the mean is included
        if self.mean is not None:
            ax = visualization.plot(self.mean, space="S2", color="red", ax=ax, s=300, alpha=1.0, label="Centroid")
        ManifoldPlot._create_legend(title ='Principal Geodesic components on Hypersphere', ax=ax)

        plt.show()
