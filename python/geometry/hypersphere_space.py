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

import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from typing import List
import numpy as np
from geometry.geometric_space import GeometricSpace, ManifoldPoint
import geomstats.backend as gs
from geometry import InformationGeometricException
__all__ = ['HypersphereSpace']


class HypersphereSpace(GeometricSpace):
    """
        Define the Hypersphere geometric space as a 2D manifold in a 3D Euclidean space.
        The key functions are:
        sample: Select uniform data point on the hypersphere
        tangent_vectors: Define a tangent vector from a vector in Euclidean space and a
                         location on the hypersphere
        show: Display the hypersphere and related components in 3D
    """

    def __init__(self, equip: bool = False, intrinsic: bool = False) -> None:
        """
            Constructor Hypersphere geometric space as a 2D manifold in a 3D Euclidean space.
            @param equip Specified that the Hypersphere instance has to be equipped, False as default
            @type equip bool
            @param intrinsic Flag to specify the coordinated as intrinsic with default as extrinsic coordinates
            @type intrinsic bool
        """
        dim = 2
        super(HypersphereSpace, self).__init__(dim, intrinsic)
        GeometricSpace.manifold_type = 'Hypersphere'

        # 1. Instantiate the Hypersphere
        self.space = Hypersphere(dim=self.dimension, equip=equip)
        # 2. Generated the default metric
        self.hypersphere_metric = HypersphereMetric(self.space)

    def belongs(self, manifold_pt: ManifoldPoint) -> bool:
        point = manifold_pt.location
        assert len(point) == 3, f'Point {point} should have 3 dimension'
        """
        Test if a point belongs to this hypersphere
        @param point defined as a list of 3 values
        @return True if the point belongs to the manifold, False otherwise
        """
        return self.space.belongs(point)

    def frechet_mean(self, manifold_pts: List[ManifoldPoint]) -> np.array:
        """
        Compute the mean of between two points on manifold.
        @param manifold_pts List of data points on a manifold with optional tangent vector and geodesic
        @type manifold_pts List[ManifoldPoint]
        @return mean value as a Numpy array
        @rtype Numpy array
        """
        assert len(manifold_pts) > 1, f'Frechet mean for hypersphere requires at least 2 manifold points'
        from geomstats.learning.frechet_mean import FrechetMean

        frechet_mean = FrechetMean(self.space)
        x = np.stack(arrays=(manifold_pts[0].location, manifold_pts[1].location), axis=0)
        frechet_mean.fit(x)
        return frechet_mean.estimate_

    def sample(self, num_samples: int) -> np.array:
        """
        Generate random data on this Hypersphere
        @param num_samples Number of sample data points on the Hypersphere
        @return Numpy array of random data points
        """
        return self.space.random_uniform(num_samples)

    def tangent_vectors(self, manifold_points: List[ManifoldPoint]) -> List[np.array]:
        """
        Compute the tangent vectors for a set of manifold point as pair
        (location, vector). The tangent vectors are computed by projection to the
        tangent plane.
        @param manifold_points List of pair (location, vector) on the manifold
        @return List of tangent vector for each location
        """
        return [self.__tangent_vector(point) for point in manifold_points]

    def geodesics(self,
                  manifold_points: List[ManifoldPoint],
                  tangent_vectors: List[np.array]) -> List[np.array]:
        """
        Compute the path (x,y,z) values for the geodesic
        @param manifold_points  Set of manifold points as pair (location, vector)
        @param tangent_vectors List of vectors associated with each location on the manifold
        @return List of geodesics as Numpy array of coordinates
        """
        return [self.__geodesic(point, tgt_vec)
                for point, tgt_vec in zip(manifold_points, tangent_vectors) if point.geodesic]

    def extrinsic_to_intrinsic(self, manifold_pts: List[ManifoldPoint]) -> List[ManifoldPoint]:
        """
        Convert the extrinsic coordinates of a list of manifold points into intrinsic coordinates
        @param manifold_pts List of manifold which coordinates/location has to be converted
        @return manifold points which location is defined as intrinsic coordinates
        """
        return [ManifoldPoint(
            id=pt.id,
            location=pt.to_intrinsic(self.space),
            tgt_vector=pt.tgt_vector,
            geodesic=pt.geodesic,
            intrinsic=True) for pt in manifold_pts]

    def intrinsic_to_extrinsic(self, manifold_pts: List[ManifoldPoint]) -> List[ManifoldPoint]:
        """
        Convert the intrinsic coordinates of a list of manifold points into extrinsic coordinates
        @param manifold_pts List of manifold which coordinates/location has to be converted
        @return manifold points which location is defined as extrinsic coordinates
        """
        return [ManifoldPoint(
            id=pt.id,
            location=pt.to_extrinsic(self.space),
            tgt_vector=pt.tgt_vector,
            geodesic=pt.geodesic,
            intrinsic=False) for pt in manifold_pts]

    def extrinsic_to_spherical(self, manifold_pts: List[ManifoldPoint]) -> List[ManifoldPoint]:
        return [ManifoldPoint(
            id=pt.id,
            location=self.space.extrinsic_to_spherical(pt.location),
            tgt_vector=pt.tgt_vector,
            geodesic=pt.geodesic,
            intrinsic=False) for pt in manifold_pts]

    def spherical_to_extrinsic(self, manifold_pts: List[ManifoldPoint]) -> List[ManifoldPoint]:
        return [ManifoldPoint(
            id=pt.id,
            location=self.space.spherical_to_extrinsic(pt.location),
            tgt_vector=pt.tgt_vector,
            geodesic=pt.geodesic,
            intrinsic=False) for pt in manifold_pts]

    def extrinsic_to_intrinsic_polar(self, manifold_pts: List[ManifoldPoint]) -> List[ManifoldPoint]:
        return [ManifoldPoint(
            id=pt.id,
            location=pt.to_intrinsic_polar(self.space),
            tgt_vector=pt.tgt_vector,
            geodesic=pt.geodesic,
            intrinsic=False) for pt in manifold_pts]

    def show_manifold(self,
                      manifold_points: List[ManifoldPoint],
                      euclidean_points: List[np.array] = None) -> None:
        """
        Display the various components on a manifold such as data points, tangent vector,
        end point (exp. map), Geodesics
        @param manifold_points  Set of manifold points as pair (id, location, tangent vector)
        @param euclidean_points Set of points in the Euclidean space
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        _ax = fig.add_subplot(111, projection="3d")
        _ax.set_facecolor('#F2F9FE')
        fig.set_facecolor('#F2F9FE')

        # Walk through the list of data point on the manifold
        for manifold_pt in manifold_points:
            ax = visualization.plot(
                manifold_pt.location,
                ax=_ax,
                space="S2",
                s=80,
                alpha=1.0,
                label=manifold_pt.id)

            # If the tangent vector has to be extracted and computed
            if manifold_pt.tgt_vector is not None:
                tgt_vec, end_pt = self.__tangent_vector(manifold_pt)
                # Show the end point
                # ax = visualization.plot(end_pt, ax=ax, space="S2", s=100, alpha=0.8, label=f'End {manifold_pt.id}')
                arrow = visualization.Arrow3D(manifold_pt.location, vector=tgt_vec)
                arrow.draw(_ax, color="red")

                # If the geodesic is to be computed and displayed
                if manifold_pt.geodesic:
                    geodesics = self.__geodesic(manifold_pt, tgt_vec)

                    # Arbitrary plot 40 data point for the geodesic from the tangent vector
                    geodesics_pts = geodesics(gs.linspace(0.0, 1.0, 40))
                    ax = visualization.plot(
                        geodesics_pts,
                        ax=_ax,
                        space="S2",
                        color='blue',
                        s=15,
                        label=f'Geodesic {manifold_pt.id}')

        if euclidean_points is not None:
            for index, euclidean_pt in enumerate(euclidean_points):
                ax.plot(
                    euclidean_pt[0],
                    euclidean_pt[1],
                    euclidean_pt[2],
                    **{'label': f'Euclidean mean', 'color': 'black'},
                    alpha=0.5)
        ax.grid()
        ax.legend()
        plt.show()

        # ------------------  Helper methods  -------------------
    @staticmethod
    def __cartesian_to_polar(c_coordinates: np.array) -> np.array:
        import math
        if len(c_coordinates) != 2:
            raise InformationGeometricException(f'Number of coordinates {len(c_coordinates)} should be 2')

        x = c_coordinates[0]
        y = c_coordinates[1]
        if x == 0.0 and y == 0.0:
            raise InformationGeometricException(f'x {x} and y {y} should not be 0')
        r = math.sqrt(x ** 2 + y ** 2)
        theta = math.acos(x/r) if y >= 0.0 else -math.acos(x/r)
        return np.array([r, theta])

    def __extrinsic_to_polar(self, c_coordinates: np.array) -> np.array:
        return HypersphereSpace.__cartesian_to_polar(self.extrinsic_to_intrinsic(c_coordinates))

    def __geodesic(self, manifold_point: ManifoldPoint, tangent_vec: np.array) -> np.array:
        return self.hypersphere_metric.geodesic(
            initial_point=manifold_point.location,
            initial_tangent_vec=tangent_vec
        )

    def __tangent_vector(self, point: ManifoldPoint) -> (np.array, np.array):
        vector = gs.array(point.tgt_vector)
        tangent_v = self.space.to_tangent(vector, base_point=point.location)
        end_point = self.hypersphere_metric.exp(tangent_vec=tangent_v, base_point=point.location)
        return tangent_v, end_point
