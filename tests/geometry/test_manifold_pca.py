import unittest

from geomstats.geometry.euclidean import Euclidean

from geometry.manifold_pca import ManifoldPCA, PrincipalComponents
from geometry.visualization.hypersphere_plot import HyperspherePlot
from geometry.visualization.manifold_plot import ManifoldPlot
import numpy as np
import logging

class TangentPCATest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_components_tangent_pca(self):
        from geomstats.geometry.hypersphere import Hypersphere

        sphere = Hypersphere(dim=2)
        X = sphere.random_uniform(n_samples=8)
        logging.info(f'Random data point on Sphere:\n{X}')
        tangent_pca = ManifoldPCA(sphere)
        principal_components, _ = tangent_pca.estimate(X)
        logging.info(f'\nPrincipal components:\n{principal_components}')

    @unittest.skip('Ignore')
    def test_components_euclidean_pca(self):
        from geomstats.geometry.euclidean import Euclidean

        sphere = Euclidean(dim=3)
        X = np.random.rand(8, 3)
        logging.info(f'Random data points on 3D:\n{X}')
        tangent_pca = ManifoldPCA(sphere)
        principal_components = tangent_pca.estimate(X)
        logging.info(f'\nPrincipal components:\n{principal_components.components}')
        self.assertTrue(len(principal_components.components) == 3)


    def test_principal_projection(self):
        from geomstats.geometry.hypersphere import Hypersphere

        sphere = Hypersphere(dim=2)
        X = sphere.random_uniform(n_samples=8)

        manifold_pca = ManifoldPCA(sphere)
        principal_components = manifold_pca.estimate(X)
        logging.info(f'\nPrincipal components for Hypersphere:\n{principal_components.components}')
        projected_points = manifold_pca.project(X)
        logging.info(f'\nProjected points on tangent space:\n{projected_points}')
        self.assertTrue(len(projected_points) == len(X))
        self.assertTrue(len(projected_points[0]) == 2)

        geodesic_components = manifold_pca.geodesics(principal_components)
        hypersphere_plot = HyperspherePlot(X, principal_components.base_point)
        hypersphere_plot.show(geodesic_components)

        TangentPCATest.euclidean_scatter(principal_components, projected_points)

        euclidean_pca = ManifoldPCA(Euclidean(3))
        principal_components = euclidean_pca.estimate(X)
        logging.info(f'\nPrincipal components in Euclidean space:\n{principal_components.components}')
        projected_points = euclidean_pca.project(X)
        logging.info(f'\nProjected points in Euclidean space:\n{projected_points}')
        TangentPCATest.euclidean_scatter(principal_components, projected_points)

    @staticmethod
    def euclidean_scatter(principal_components: PrincipalComponents, projected_points: np.array) -> None:
        import matplotlib.pyplot as plt
        dim = projected_points.shape[1]
        font_dict = {'family': 'sans-serif', 'color': 'blue', 'weight': 'bold', 'size': 16}

        if dim == 2:
            x = projected_points[:, 0]
            y = projected_points[:, 1]
            for idx in range(len(x)):
                plt.scatter(x[idx], y[idx], cmap='cool', s=120, marker='o', label=f'data {idx}')

            plt.scatter(principal_components.base_point[0],
                        principal_components.base_point[1],
                        s=260,
                        color='red',
                        marker='o',
                        label='Frechet centroid')
            plt.title(label='Projection on 2D Hypersphere tangent', fontdict=font_dict)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid()
            plt.legend()
        else:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")

            x = projected_points[:, 0]
            y = projected_points[:, 1]
            z = projected_points[:, 2]
           # sc = ax.scatter(x, y, z, c=x, cmap='cool', s=120, marker='o', label='Euclidean')
            # fig.colorbar(sc, ax=ax, label='Z-axis Value')
            for idx in range(len(x)):
                ax.scatter(x[idx], y[idx], z[idx], cmap='cool', s=120, marker='o', label=f'data {idx}')


            ax.scatter(principal_components.base_point[0],
                       principal_components.base_point[1],
                       principal_components.base_point[2],
                       s=260,
                       color='red',
                       marker='o',
                       label='Arithmetic mean')
            ManifoldPlot._create_legend(title='Projection with linear PCA', ax=ax)
        plt.show()


