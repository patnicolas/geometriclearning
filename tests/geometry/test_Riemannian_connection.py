import unittest
from geomstats.geometry.hypersphere import Hypersphere
import numpy as np
from geometry.Riemannian_connection import RiemannianConnection
from geometry.manifold_point import ManifoldPoint
import logging
import python


class TestRiemannianConnection(unittest.TestCase):

    def test_pprint(self):
        from pprint import pprint

        base_pt = np.array([0.4, 0.1, 0.0])
        manifold_base_pt = ManifoldPoint(id='base', location=base_pt, tgt_vector=[0.5, 0.1, 2.1])
        pprint(manifold_base_pt)
        logging.info(f'My manifold: {manifold_base_pt}')

    def test_init(self):
        dim = 2
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True)
        riemannian_connection = RiemannianConnection(hypersphere, manifold_type='HyperSphere')
        logging.info(str(riemannian_connection))

    def test_inner_product_identity(self):
        dim = 2
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True)
        riemannian_connection = RiemannianConnection(hypersphere, manifold_type='HyperSphere')
        vector1 = np.array([0.4, 0.1, 0.8])
        base_point = np.array([0.4, 0.1, 0.0])
        inner_product = riemannian_connection.inner_product(vector1, vector1, base_point)
        self.assertAlmostEqual(first=inner_product, second=0.81, places=None, msg=None, delta=0.0001)
        logging.info(f'Inner product tangent vector: {inner_product}')
        euclidean_inner_product = RiemannianConnection.euclidean_inner_product(vector1, vector1)
        logging.info(f'\n{euclidean_inner_product=}')

    def test_inner_product(self):
        dim = 2
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True)
        riemannian_connection = RiemannianConnection(hypersphere, manifold_type='HyperSphere')
        vector1 = np.array([0.4, 0.1, 0.8])
        vector2 = np.array([-0.4, -0.1, -0.8])
        base_point = np.array([0.4, 0.1, 0.0])
        inner_product = riemannian_connection.inner_product(vector1, vector2, base_point)
        self.assertAlmostEqual(first=inner_product, second=-0.81, places=None, msg=None, delta=0.0001)
        logging.info(f'{inner_product=}')

    def test_norm(self):
        dim = 2
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True)
        riemannian_connection = RiemannianConnection(hypersphere, manifold_type='HyperSphere')
        vector = np.array([0.4, 0.1, 0.8])
        base_point = np.array([0.4, 0.1, 0.0])
        norm = riemannian_connection.norm(vector, base_point)
        self.assertAlmostEqual(first=norm, second=np.linalg.norm(vector), places=None, msg=None, delta=0.0001)
        logging.info(f'{norm=}')

    def test_parallel_transport(self):
        dim = 2
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True)
        riemann_connection = RiemannianConnection(hypersphere, manifold_type='HyperSphere')
        base_pt = np.array([1.5, 2.0, 1.6])
        manifold_base_pt = ManifoldPoint(id='base', location=base_pt, tgt_vector=[0.5, 0.1, 2.1])
        manifold_end_pt = ManifoldPoint(id='end', location=base_pt + 0.4)
        parallel_transport = riemann_connection.parallel_transport(manifold_base_pt, manifold_end_pt)
        logging.info(f'{parallel_transport=}')

    def test_levi_civita_coefficients(self):
        dim = 2
        # Instantiate the Hypersphere
        hypersphere = Hypersphere(dim=dim, equip=True)
        riemann_connection = RiemannianConnection(hypersphere, manifold_type='HyperSphere')
        base_pt = np.array([1.5, 2.0, 1.6])
        u = np.arctan(1.0/0.07091484)
        logging.info(f'U:{u}')
        v = 0.5*np.sin(2*u)
        logging.info(f'V:{v}')
        levi_civita_coefficients = riemann_connection.levi_civita_coefficients(base_pt)
        logging.info(f'{levi_civita_coefficients=}')


    def test_curvature_tensor(self):
        hypersphere = Hypersphere(dim=2, equip=True, intrinsic=True)
        riemann_connection = RiemannianConnection(hypersphere, manifold_type='HyperSphere')
        base_pt = np.array([1.5, 2.0, 1.6])
        X = np.array([0.4, 0.1, 0.8])
        Y = np.array([0.7, 0.1, -0.2])
        Z = np.array([0.4, 0.9, 0.0])
        curvature = riemann_connection.curvature_tensor([X, Y, Z], base_pt)
        logging.info(f'{curvature=}')

    def test_curvature_derivative_tensor(self):
        hypersphere = Hypersphere(dim=2, equip=True, intrinsic=True)
        riemann_connection = RiemannianConnection(hypersphere, manifold_type='HyperSphere')
        base_pt = np.array([0.5, 1.9, 0.4])
        X = np.array([0.4, 0.1, 0.8])
        Y = np.array([0.4, 0.3, 0.8])
        Z = np.array([0.4, 0.6, 0.8])
        T = np.array([0.4, -0.5, 0.8])
        curvature_derivative = riemann_connection.curvature_derivative_tensor([X, Y, Z, T], base_pt)
        logging.info(f'{curvature_derivative=}')

    def test_sectional_curvature_tensor(self):
        hypersphere = Hypersphere(dim=2, equip=True, intrinsic=True)
        riemann_connection = RiemannianConnection(hypersphere, manifold_type='HyperSphere')
        base_pt = np.array([1.5, 2.0, 1.6])

        X = np.array([0.4, 0.1, 0.8])
        Y = np.array([0.4, 0.1, -0.8])
        sectional_curvature = riemann_connection.sectional_curvature_tensor(X, Y, base_pt)
        logging.info(f'{sectional_curvature=}')

        X = np.array([0.4, 0.1, 0.8])
        Y = np.array([-0.4, -0.1, -0.8])
        sectional_curvature = riemann_connection.sectional_curvature_tensor(X, Y, base_pt)
        logging.info(f'{sectional_curvature=}')

        X = np.array([0.4, 0.1, 0.8])
        Y = np.array([0.8, 0.2, 1.6])
        sectional_curvature = riemann_connection.sectional_curvature_tensor(X, Y, base_pt)
        logging.info(f'{sectional_curvature=}')


if __name__ == '__main__':
    unittest.main()
