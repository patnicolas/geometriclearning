__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.manifold import Manifold
import numpy as np
from typing import Optional, List, AnyStr
from geometry import GeometricException
from dataclasses import dataclass


@dataclass
class TangentComponent:
    base_point: np.array
    principal_components: np.array

    def __str__(self) -> AnyStr:
        return f'\nBase point:\n{self.base_point}\nComponents: {self.principal_components}'


class TangentPCA(object):
    def __init__(self, space: Manifold) -> None:
        """
        Constructor for PCA on Tangent of manifold. The constructor instantiates the Geomstats TangentPCA class
        @param space: Manifold under consideration
        @type space: Manifold
        """
        from geomstats.learning.pca import TangentPCA
        self.tgt_pca = TangentPCA(space, n_components=2)
        self.space = space

    def estimate(self, manifold_pts: np.array, base_point: Optional[np.array] = None) -> TangentComponent:
        """
        Compute the principal components on the tangent space of a manifold at a given base point. If the base
        point is not provided, the Frechet mean is used
        @param manifold_pts: Points on the data manifold
        @type manifold_pts: Numpy array
        @param base_point: Optional base point on the manifold that anchors the tangent space
        @type base_point: Numpy array
        @return: An instance of TangentComponent, with a base_point and principal components
        @rtype: TangentComponent
        """
        from geomstats.learning.frechet_mean import FrechetMean

        # If no
        if base_point is None:
            frechet_mean = FrechetMean(self.space)
            frechet_mean.fit(X=manifold_pts, y=None)
            base_point = frechet_mean.estimate_

        self.tgt_pca.fit(manifold_pts, base_point=base_point)
        return TangentComponent(base_point, self.tgt_pca.components_)

    def project(self, manifold_pts: np.array) -> np.array:
        """
        Project the original data points to the tangent plane using the transformation from eigenvectors
        @param manifold_pts: List of points on the manifold
        @type manifold_pts: Numpy array
        @return: Numpy arrays of points on the tangent space
        @rtype: Numpy array
        """
        return self.tgt_pca.transform(manifold_pts)

    def geodesics(self, tangent_component: TangentComponent) -> List[np.array]:
        """
        Generate the geodesics on a Hypersphere that correspond to the principal components at a given base point
        on the hypersphere.
        @param tangent_component: Tangent component {base_point on manifold, Principal components on tangent space)
        @type tangent_component: TangentComponent
        @return: List of geodesic values representing the principal components
        @rtype: List of Numpy arrays
        """
        import geomstats.backend as gs

        if not isinstance(self.space, Hypersphere):
            raise GeometricException(f'Cannot extract geodesics from this type of manifold {str(self.space)}')

        base_point = tangent_component.base_point
        components = tangent_component.principal_components
        geodesics = [self.space.metric.geodesic(initial_point=base_point, initial_tangent_vec=component)
                     for component in components]

        trace = gs.linspace(-1.0, 1.0, 200)
        geodesic_traces = [geodesic(trace) for geodesic in geodesics]
        return geodesic_traces





