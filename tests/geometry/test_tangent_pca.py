import unittest

from geometry.tangent_pca import TangentPCA
from geometry.visualization.hypersphere_plot import HyperspherePlot


class TangentPCATest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_principal_components(self):
        from geomstats.geometry.hypersphere import Hypersphere

        sphere = Hypersphere(dim=2)
        X = sphere.random_uniform(n_samples=8)
        print(f'Random data point on Sphere:\n{X}')
        tangent_pca = TangentPCA(sphere)
        principal_components, _ = tangent_pca.estimate(X)
        print(f'\nPrincipal components:\n{principal_components}')

    def test_principal_projection(self):
        from geomstats.geometry.hypersphere import Hypersphere

        sphere = Hypersphere(dim=2)
        X = sphere.random_uniform(n_samples=8)
        tangent_pca = TangentPCA(sphere)
        tangent_components = tangent_pca.estimate(X)
        print(f'\nPrincipal components:\n{tangent_components.principal_components}')
        projected_points = tangent_pca.project(X)
        print(f'\nProjected points:\n{projected_points}')
        self.assertTrue(len(projected_points) == len(X))
        self.assertTrue(len(projected_points[0]) == 2)

        geodesic_components = tangent_pca.geodesics(principal_components=principal_components, base_point=base_point)
        hypersphere_plot = HyperspherePlot(X, base_point)
        hypersphere_plot.show(geodesic_components)
