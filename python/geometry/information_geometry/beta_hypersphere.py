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


# 3rd Party imports
import geomstats.backend as gs
import matplotlib.pyplot as plt
# Library imports
from geometry.information_geometry.geometric_distribution import GeometricDistribution
__all__ = ['BetaHypersphere']


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
        assert num_manifold_pts > 1, f'Number of manifold points {num_manifold_pts} should be > 1'
        assert num_interpolations > 1, f'Number of interpolation {num_interpolations} should be > 1'

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
