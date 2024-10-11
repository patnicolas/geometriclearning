import unittest
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
import numpy as np
from manifolds.Riemannianconnection import RiemannianConnection
from manifolds.manifoldpoint import ManifoldPoint



class TestRiemannianConnection(unittest.TestCase):

    def test_pprint(self):
        from pprint import pprint

        base_pt = np.array([0.4, 0.1, 0.0])
        manifold_base_pt = ManifoldPoint(
            id='base',
            location=base_pt,
            tgt_vector=np.array([[0.5, 0.1, 2.1]]))
        pprint(manifold_base_pt)
        print(f'My manifold: {manifold_base_pt}')

    def test_init(self):
        dim = 2
        coordinates_type = 'extrinsic'
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True, default_coords_type=coordinates_type)
        riemannian_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        print(str(riemannian_connection))

    def test_inner_product_identity(self):
        dim = 2
        coordinates_type = 'extrinsic'
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True, default_coords_type=coordinates_type)
        riemannian_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        vector1 = np.array([0.4, 0.1, 0.8])
        base_point = np.array([0.4, 0.1, 0.0])
        inner_product = riemannian_connection.inner_product(vector1, vector1, base_point)
        self.assertAlmostEqual(first=inner_product, second=0.81, places=None, msg=None, delta=0.0001)
        print(f'Inner product tangent vector: {inner_product}')
        euclidean_inner_product = RiemannianConnection.euclidean_inner_product(vector1, vector1)
        print(f'Euclidean inner product: {euclidean_inner_product}')

    def test_inner_product(self):
        dim = 2
        coordinates_type = 'extrinsic'
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True, default_coords_type=coordinates_type)
        riemannian_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        vector1 = np.array([0.4, 0.1, 0.8])
        vector2 = np.array([-0.4, -0.1, -0.8])
        base_point = np.array([0.4, 0.1, 0.0])
        inner_product = riemannian_connection.inner_product(vector1, vector2, base_point)
        self.assertAlmostEqual(first = inner_product, second=-0.81, places=None, msg=None, delta=0.0001)
        print(f'Inner product: {inner_product}')

    def test_norm(self):
        dim = 2
        coordinates_type = 'extrinsic'
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True, default_coords_type=coordinates_type)
        riemannian_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        vector = np.array([0.4, 0.1, 0.8])
        base_point = np.array([0.4, 0.1, 0.0])
        norm = riemannian_connection.norm(vector, base_point)
        self.assertAlmostEqual(first=norm, second=np.linalg.norm(vector), places=None, msg=None, delta=0.0001)
        print(f'Norm: {norm}')

    def test_parallel_transport(self):
        dim = 2
        coordinates_type = 'extrinsic'
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True, default_coords_type=coordinates_type)
        riemann_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        base_pt = np.array([1.5, 2.0, 1.6])
        manifold_base_pt = ManifoldPoint(
            id='base',
            location=base_pt,
            tgt_vector=np.array([[0.5, 0.1, 2.1]]))
        manifold_end_pt = ManifoldPoint(id='end',location=base_pt + 0.4)
        parallel_trans = riemann_connection.parallel_transport(manifold_base_pt, manifold_end_pt)
        print(f'Parallel transport: {parallel_trans}')

    def test_levi_civita_coefficients(self):
        dim = 2
        coordinates_type = 'extrinsic'
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True, default_coords_type=coordinates_type)
        riemann_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        base_pt = np.array([1.5, 2.0, 1.6])
        u = np.arctan(1.0/0.07091484)
        print(f'U:{u}')
        v = 0.5*np.sin(2*u)
        print(f'V:{v}')
        levi_civita_coefs = riemann_connection.levi_civita_coefficients(base_pt)
        print(f'Levi-Civita coefficients:\n{str(levi_civita_coefs)}')


    def test_curvature_tensor(self):
        hypersphere = Hypersphere(dim=2, equip=True, default_coords_type='intrinsic')
        riemann_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        base_pt = np.array([1.5, 2.0, 1.6])
        X = np.array([0.4, 0.1, 0.8])
        Y = np.array([0.7, 0.1, -0.2])
        Z = np.array([0.4, 0.9, 0.0])
        curvature = riemann_connection.curvature_tensor([X, Y, Z], base_pt)
        print(f'Curvature: {curvature}')

    def test_curvature_derivative_tensor(self):
        hypersphere = Hypersphere(dim=2, equip=True, default_coords_type='intrinsic')
        riemann_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        base_pt = np.array([0.5, 1.9, 0.4])
        X = np.array([0.4, 0.1, 0.8])
        Y = np.array([0.4, 0.3, 0.8])
        Z = np.array([0.4, 0.6, 0.8])
        T = np.array([0.4, -0.5, 0.8])
        curvature_derivative = riemann_connection.curvature_derivative_tensor([X, Y, Z, T], base_pt)
        print(f'Curvature derivative: {curvature_derivative}')

    def test_sectional_curvature_tensor(self):
        hypersphere = Hypersphere(dim=2, equip=True, default_coords_type='intrinsic')
        riemann_connection = RiemannianConnection(hypersphere, 'HyperSphere')
        base_pt = np.array([1.5, 2.0, 1.6])

        X = np.array([0.4, 0.1, 0.8])
        Y = np.array([0.4, 0.1, -0.8])
        curvature = riemann_connection.sectional_curvature_tensor(X, Y, base_pt)
        print(f'Sectional curvature: {curvature}')

        X = np.array([0.4, 0.1, 0.8])
        Y = np.array([-0.4, -0.1, -0.8])
        curvature = riemann_connection.sectional_curvature_tensor(X, Y, base_pt)
        print(f'Sectional curvature: {curvature}')

        X = np.array([0.4, 0.1, 0.8])
        Y = np.array([0.8, 0.2, 1.6])
        curvature = riemann_connection.sectional_curvature_tensor(X, Y, base_pt)
        print(f'Sectional curvature: {curvature}')

if __name__ == '__main__':
    unittest.main()