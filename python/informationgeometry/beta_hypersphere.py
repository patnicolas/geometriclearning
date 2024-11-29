__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from informationgeometry.geometric_distribution import GeometricDistribution
import geomstats.backend as gs
import matplotlib.pyplot as plt


class BetaHypersphere(GeometricDistribution):
    def __init__(self) -> None:
        """
        Constructor for the Normal Distribution on a Hypersphere
        """
        from geomstats.information_geometry.beta import BetaDistributions

        super(BetaHypersphere, self).__init__()
        self.beta = BetaDistributions()

    def show_distribution(self, num_manifold_pts: int, num_interpolations: int) -> bool:
        """
        Display the Beta distribution for multiple random points on a hypersphere. The data points are
        randomly generated using the Von-mises random generator.
        @param num_manifold_pts: Number of data points on the hypersphere
        @type num_manifold_pts: int
        @param num_interpolations: Number of interpolation points to draw the Beta distributions
        @type num_interpolations: int
        @return: True if number of Beta density functions displayed is correct, False else
        @rtype: bool
        """
        # Generate random points on Hypersphere using Von Mises algorithm
        manifold_pts = self._random_manifold_points(num_manifold_pts)
        t = gs.linspace(0, 1.1, num_interpolations)[1:]
        # Define the beta pdfs associated with each
        beta_values_pdfs = [self.beta.point_to_pdf(manifold_pt.location)(t) for manifold_pt in manifold_pts]

        # Generate, normalize and display each Beta distribution
        for beta_values in beta_values_pdfs:
            min_beta = min(beta_values)
            delta_beta = max(beta_values) - min_beta
            y = [(beta_value - min_beta)/delta_beta  for beta_value in beta_values]
            plt.plot(t, y)
        plt.title(f'Beta distribution on Hypersphere')
        plt.show()

        return len(beta_values_pdfs) == num_manifold_pts
