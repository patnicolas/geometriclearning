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

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.manifold import Manifold
import numpy as np
from typing import Optional, List, AnyStr
from geometry import GeometricException
from dataclasses import dataclass
__all__ = ['PrincipalComponents', 'ManifoldPCA']


@dataclass
class PrincipalComponents:
    base_point: np.array
    components: np.array

    def __str__(self) -> AnyStr:
        return f'\nBase point:\n{self.base_point}\nComponents: {self.components}'


class ManifoldPCA(object):
    def __init__(self, space: Manifold) -> None:
        """
        Constructor for PCA on Tangent of manifold if the space is not defined as Euclidean, linear PCA otherwise.
        The constructor instantiates the Geomstats TangentPCA class for manifold and sklearn PCA for linear PCA
        @param space: Manifold under consideration (Hypersphere, Euclidean)
        @type space: Manifold
        """
        from geomstats.learning.pca import TangentPCA
        from geomstats.geometry.euclidean import Euclidean
        from sklearn.decomposition import PCA

        self.pca = TangentPCA(space, n_components=2) if not isinstance(space, Euclidean) else PCA(n_components=3)
        self.space = space

    def estimate(self,
                 manifold_pts: np.array,
                 base_point: Optional[np.array] = None) -> PrincipalComponents:
        """
        Compute the principal components on the tangent space of a manifold at a given base point if the manifold
        is of type TangentPCA, Euclidean PCA otherwise
        The Frechet mean is used for PCA on tangent space if base point is not provided
        The arithmetic mean is used for PCA in Euclidean space if base point is not provided

        @param manifold_pts: Points on the data manifold
        @type manifold_pts: Numpy array
        @param base_point: Optional base point on the manifold that anchors the tangent space
        @type base_point: Numpy array
        @return: An instance of TangentComponent, with a base_point and principal components
        @rtype: PrincipalComponents
        """
        from geomstats.geometry.euclidean import Euclidean
        return self.__tangent_pca(manifold_pts, base_point) if not isinstance(self.space, Euclidean)  \
            else self.__euclidean_pca(manifold_pts, base_point)

    def project(self, manifold_pts: np.array) -> np.array:
        """
        Project the original data points to the tangent plane using the transformation from eigenvectors
        @param manifold_pts: List of points on the manifold
        @type manifold_pts: Numpy array
        @return: Numpy arrays of points on the tangent space
        @rtype: Numpy array
        """
        return self.pca.transform(manifold_pts)

    def geodesics(self, tangent_component: PrincipalComponents) -> List[np.array]:
        """
        Generate the geodesics on a Hypersphere that correspond to the principal components at a given base point
        on the hypersphere.
        @param tangent_component: Tangent component {base_point on manifold, Principal components on tangent space)
        @type tangent_component: PrincipalComponents
        @return: List of geodesic values representing the principal components
        @rtype: List of Numpy arrays
        """
        import geomstats.backend as gs

        # For now, we generate geodesics only for spheres
        if not isinstance(self.space, Hypersphere):
            raise GeometricException(f'Cannot extract geodesics from this type of manifold {str(self.space)}')

        # Extract the geodesic for each of the principal components
        base_point = tangent_component.base_point
        components = tangent_component.components
        geodesics = [self.space.metric.geodesic(initial_point=base_point, initial_tangent_vec=component)
                     for component in components]

        # Generate a trace of 200 data points for visualization purpose
        trace = gs.linspace(-1.0, 1.0, 200)
        geodesic_traces = [geodesic(trace) for geodesic in geodesics]
        return geodesic_traces

    """ -------------------------  Private supporting methods --------------------- """

    def __tangent_pca(self,
                      manifold_pts: np.array,
                      base_point: Optional[np.array] = None) -> PrincipalComponents:

        from geomstats.learning.frechet_mean import FrechetMean

        # If no base points have been defined ... then use the centroid
        if base_point is None:
            frechet_mean = FrechetMean(self.space)
            frechet_mean.fit(X=manifold_pts, y=None)
            base_point = frechet_mean.estimate_
        # Compute the 2-dimension principal components in intrinsic coordinates
        self.pca.fit(manifold_pts, base_point=base_point)
        return PrincipalComponents(base_point, self.pca.components_)

    def __euclidean_pca(self,
                        manifold_pts: np.array,
                        base_point: Optional[np.array] = None) -> PrincipalComponents:
        self.pca.fit(manifold_pts)

        mean = np.mean(manifold_pts, axis=0) if base_point is None else base_point
        return PrincipalComponents(mean, self.pca.components_)




