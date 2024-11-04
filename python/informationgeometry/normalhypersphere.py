__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from informationgeometry.geometricdistribution import GeometricDistribution
import geomstats.backend as gs
import matplotlib.pyplot as plt

"""
Define a Normal Distribution on an Hypersphere using the Geomstats Python library
The purpose of this class is to display variants of two Normal distribution on a Hypersphere
@see informationgeometry.GeometricDistribution
This implementation relies on the manifold point defined in manifolds.ManifoldPoint
"""


class NormalHypersphere(GeometricDistribution):
    def __init__(self) -> None:
        """
        Constructor for the Normal Distribution on a Hypersphere
        """
        from geomstats.information_geometry.normal import NormalDistributions

        super(NormalHypersphere, self).__init__()
        self.normal = NormalDistributions(sample_dim=1)

    def show_distribution(self, num_pdfs: int, num_manifold_pts: int) -> bool:
        """
        Display the normal distribution for two points on a hypersphere. The data points are
        randomly generated using the Von-mises random generator.
        @param num_pdfs: Number of density functions to be displayed
        @type num_pdfs: int
        @param num_manifold_pts: Number of interpolation points on geodesic between the two data points on the manifold
        @type num_manifold_pts: int
        @return: True if number of distributions is correct, False otherwise
        @rtype: bool
        """
        manifold_pts = self._random_manifold_points(num_manifold_pts)
        # Apply the Fisher metric for the two manifold points on a Hypersphere
        geodesic_ab_fisher = self.normal.metric.geodesic(manifold_pts[0].location, manifold_pts[1].location)
        t = gs.linspace(0, 1, 100)

        # Generate the various density function associated to the Fisher metric between the
        # two point on the hypersphere
        pdfs = self.normal.point_to_pdf(geodesic_ab_fisher(t))
        x = gs.linspace(0.2, 0.7, num_pdfs)
        for i in range(num_pdfs):
            plt.plot(x, pdfs(x)[i, :]/20.0)   # Normalization factor
        plt.title(f'Normal distribution on Hypersphere')
        plt.show()
        return pdfs == num_pdfs
