__author__ = "Patrick Nicolas"
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

from information_geometry.geometric_distribution import GeometricDistribution
import geomstats.backend as gs
import matplotlib.pyplot as plt
__all__ = ['NormalHypersphere']


class NormalHypersphere(GeometricDistribution):
    """
    Define a Normal Distribution on an Hypersphere using the Geomstats Python library
    The purpose of this class is to display variants of two Normal distribution on a Hypersphere
    @see information_geometry.GeometricDistribution
    This implementation relies on the manifold point defined in manifolds.ManifoldPoint
    """

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
