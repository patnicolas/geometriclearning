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

# Standard Library imports
from typing import AnyStr, List
# 3rd Party imports
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
import numpy as np
from dataclasses import dataclass
import geomstats.backend as gs
__all__ = ['KMeansCluster', 'KMeansOnManifold']

@dataclass(frozen=True)
class KMeansCluster:
    """
    Data class to collect result associated with a cluster extracted with k-means unsupervised training
    @param center: Centroid
    @type center: Numpy array (3-dimension)
    @param label: Identifier for the cluster
    @type label: Numpy array
    """
    center: np.array
    label: np.array

    def __str__(self) -> AnyStr:
        return f'Center: {self.center}, Label: {self.label}'


class KMeansOnManifold(object):
    """
    Class to wrap the evaluation of k-means algorithm on Euclidean space using Scikit-learn library and
    on the hypersphere (as Riemann manifold) using Geomstats library.

    The evaluation relies on synthetic clustered data on hypersphere following the following steps:
    1. Generate a template cluster by employing a random generator on the manifold.
    2. Generate 4 clusters from the template using a special orthogonal lie group in 3-dimensional space, SO(3).

    The synthetic dataset is created through random values generators on the hypersphere following these distributions:
    - Uniform distribution
    - Uniform distribution with constraints
    - von Mises-Fisher distribution
    """
    random_von_mises_fisher_label = 'random_von_mises_fisher'
    random_rand_riemann_normal_label = 'random_riemann_normal'
    random_uniform_label = 'random_uniform'
    clustered_random_uniform_label = 'constrained_random_uniform'

    def __init__(self, num_samples: int, num_clusters: int, random_gen: AnyStr = 'random_von_mises_fisher'):
        """
        Constructor for the evaluation of k-means algorithm on Euclidean space and Riemann manifold (Hypersphere)
        @param num_samples: Number of random samples
        @type num_samples: int
        @param num_clusters: Number of clusters used in KMeans
        @type num_clusters: int
        @param random_gen: Random generator identifier on the hypersphere
        @type random_gen: AnyStr
        """
        assert 0 < num_samples <= 19384, f'Number of samples {num_samples} should be [1, 19384]'
        assert 0 < num_clusters <= 1024, f'Number of clusters {num_clusters} should be [1, 1024]'

        # Step 1: Initialize the manifold
        self.hypersphere = Hypersphere(dim=2, equip=True)

        # Step 2: Generate a single cluster with random data points on hypersphere
        self.random_gen = random_gen
        match random_gen:
            case 'random_von_mises_fisher':
                # Select a pivot or mean value
                _cluster = self.hypersphere.random_uniform(n_samples=2)
                # Generate the cluster
                cluster = self.hypersphere.random_von_mises_fisher(mu=_cluster[0], kappa=60, n_samples=num_samples, max_iter=200)
            case 'random_riemann_normal':
                cluster = self.hypersphere.random_riemannian_normal(n_samples=num_samples, max_iter=300)
            case 'random_uniform':
                cluster = self.hypersphere.random_uniform(n_samples=num_samples)
            case 'constrained_random_uniform':
                # Generate random values with constrains on each dimension [-1, -0.35], [0.3, 1] and [-1, 0.4]
                y = [x for x in self.hypersphere.random_uniform(n_samples=100000)
                     if x[0] <= -0.35 and x[1] >= 0.3 and x[2] <= -0.40]
                cluster = np.array(y)[0:num_samples]
            case _:
                raise ValueError(f'{random_gen} generator is not supported')
                cluster = self.hypersphere.random_uniform(n_samples=num_samples)

        # Step 3: Generate other clusters using SO(3) manifolds
        # Use the symmetric rotation lie group in dimension 3
        so3_lie_group = SpecialOrthogonal(3, equip=False)
        # Generate the clusters
        self.clusters = [cluster @ so3_lie_group.random_uniform() for _ in range(num_clusters)]

    def __str__(self) -> AnyStr:
        return '\n'.join([f'Cluster #{idx+1} {str(cluster)}' for idx, cluster in enumerate(self.clusters)])

    def euclidean_clustering(self) -> List[KMeansCluster]:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=len(self.clusters), init='k-means++', algorithm='elkan', max_iter=140)
        data = np.concatenate(self.clusters, axis=0)
        kmeans.fit(data)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        return [KMeansCluster(center, label) for center, label in zip(centers, labels)]

    def riemannian_clustering(self) -> List[KMeansCluster]:
        from geomstats.learning.kmeans import RiemannianKMeans

        self.visualize()
        kmeans = RiemannianKMeans(space=self.hypersphere, n_clusters=len(self.clusters))
        data = gs.concatenate(self.clusters, axis =0)
        kmeans.fit(data)
        centers = kmeans.centroids_
        labels = kmeans.labels_
        return [KMeansCluster(center, label) for center, label in zip(centers, labels)]

    def visualize(self):
        import matplotlib.pyplot as plt
        import geomstats.visualization as visualization

        fig = plt.figure(figsize=(8, 6))
        colors = ["red", "blue", 'black', 'green']
        data = gs.concatenate(self.clusters, axis =0)
        ax = visualization.plot(data, space="S2", marker=".", color="grey")
        for idx, cluster in enumerate(self.clusters):
            ax = visualization.plot(
               cluster , space="S2", color=colors[idx], alpha=0.7, label=f'Cluster {idx}'
            )

        ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
        ax.legend();
        ax.set_title(f'Kmeans on Hypersphere for {self.random_gen} generator')
        ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
        plt.show()

    def evaluate(self) -> np.array:
        from geomstats.learning.kmeans import RiemannianKMeans
        kmeans = RiemannianKMeans(space=self.hypersphere, n_clusters=len(self.clusters))
        data = gs.concatenate(self.clusters, axis =0)
        kmeans.fit(data)
        predicted = kmeans.predict(self.clusters[0][0:20])
        return predicted


