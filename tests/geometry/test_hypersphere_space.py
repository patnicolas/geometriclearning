import unittest
import logging
from geometry.hypersphere_space import HypersphereSpace
from geometry.manifold_point import ManifoldPoint
import os
import python
from python import SKIP_REASON


class TestGeometricSpace(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_sample_hypersphere(self):
        num_samples = 180
        manifold = HypersphereSpace()
        hypersphere_data = manifold.sample(num_samples)
        logging.info(f'\n{hypersphere_data=}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_hypersphere(self):
        num_samples = 8
        style = {'color': 'red', 'linestyle': '--', 'label': 'Edges'}
        manifold = HypersphereSpace()
        logging.info(str(manifold))
        manifold_data = manifold.sample(num_samples)
        manifold.show_manifold(manifold_points=manifold_data)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_tangent_vector(self):
        from geometry.geometric_space import GeometricSpace

        filename = '../../../data/hypersphere_data_1.txt'
        data = GeometricSpace.load_csv(filename)
        manifold = HypersphereSpace(True)
        samples = manifold.sample(8)
        manifold_points = [
            ManifoldPoint(
                id=f'data{index}',
                location=sample,
                tgt_vector=[0.5, 0.3, 0.5],
                geodesic=False) for index, sample in enumerate(samples)
        ]
        manifold = HypersphereSpace(True)
        exp_map = manifold.tangent_vectors(manifold_points)
        for vec, end_point in exp_map:
            logging.info(f'Tangent vector: {vec} End point: {end_point}')
        manifold.show_manifold(manifold_points)

    def test_show_tangent_vector_geodesics(self):
        manifold = HypersphereSpace(True)

        for _ in range(8):
            samples = manifold.sample(4)
            manifold_points = [
                ManifoldPoint(
                    id=f'data{index}',
                    location=sample,
                    tgt_vector=[-0.2, 0.3, 0.5],
                    geodesic=True) for index, sample in enumerate(samples)
            ]
            manifold.show_manifold(manifold_points)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_euclidean_mean(self):
        manifold = HypersphereSpace(True)
        samples = manifold.sample(3)
        manifold_points = [
            ManifoldPoint(
                id=f'data{index}',
                location=sample) for index, sample in enumerate(samples)
        ]
        mean = manifold.euclidean_mean(manifold_points)
        logging.info(mean)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_frechet_mean(self):
        manifold = HypersphereSpace(True)
        samples = manifold.sample(2)
        assert(manifold.belongs(samples[0]))   # Is True
        vector = [0.8, 0.4, 0.7]

        manifold_points = [
            ManifoldPoint(
                id=f'data{index}',
                location=sample,
                tgt_vector=vector,
                geodesic=False) for index, sample in enumerate(samples)
        ]
        euclidean_mean = manifold.euclidean_mean(manifold_points)
        manifold.belongs(euclidean_mean)   # Is False
        exp_map = manifold.tangent_vectors(manifold_points)
        tgt_vec, end_point = exp_map[0]
        assert manifold.belongs(end_point)     # Is True
        frechet_mean = manifold.frechet_mean(manifold_points)
        logging.info(f'Euclidean mean: {euclidean_mean}\nFrechet mean: {frechet_mean}')
        assert manifold.belongs(frechet_mean)

        frechet_pt = ManifoldPoint(
            id='Frechet mean',
            location=frechet_mean,
            tgt_vector=[0.0, 0.0, 0.0],
            geodesic=False)
        manifold_points.append(frechet_pt)
        manifold.show_manifold(manifold_points, [euclidean_mean])

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_extrinsic_to_intrinsic(self):
        intrinsic = False
        manifold = HypersphereSpace(True, intrinsic)
        # Create Manifold points with default extrinsic coordinates
        random_samples = manifold.sample(2)
        intrinsic = False
        manifold_pts = [
            ManifoldPoint(f'id{index}', value, None, False, intrinsic)
            for index, value in enumerate(random_samples)
        ]
        assert manifold.belongs(manifold_pts[0])
        logging.info(f'From extrinsic Coordinates:\n{[m_pt.location for m_pt in manifold_pts]}')
        intrinsic_manifold_pts = manifold.extrinsic_to_intrinsic(manifold_pts)
        logging.info(f'To intrinsic Coordinates: {[m_pt.location for m_pt in intrinsic_manifold_pts]}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_intrinsic_to_extrinsic(self):
        intrinsic = True
        manifold = HypersphereSpace(True, intrinsic)
        # Create Manifold points with default extrinsic coordinates
        random_samples = manifold.sample(2)
        manifold_pts = [
            ManifoldPoint(f'id{index}', value, None, False, intrinsic) for index, value in enumerate(random_samples)
        ]
        logging.info(f'From intrinsic Coordinates:\n{[m_pt.location for m_pt in manifold_pts]}')
        extrinsic_manifold_pts = manifold.intrinsic_to_extrinsic(manifold_pts)
        logging.info(f'To extrinsic Coordinates:\n{[m_pt.location for m_pt in extrinsic_manifold_pts]}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_reciprocate_coordinates(self):
        intrinsic = False
        manifold = HypersphereSpace(True)
        # Create Manifold points with default extrinsic coordinates
        random_samples = manifold.sample(2)
        manifold_pts = [
            ManifoldPoint(f'id{index}', value, None, False, intrinsic)
            for index, value in enumerate(random_samples)
        ]
        assert manifold.belongs(manifold_pts[0])
        logging.info(f'Original extrinsic coordinates:\n{[m_pt.location for m_pt in manifold_pts]}')
        intrinsic_manifold_pts = manifold.extrinsic_to_intrinsic(manifold_pts)
        logging.info(f'Intrinsic Coordinates:\n{[m_pt.location for m_pt in intrinsic_manifold_pts]}')
        extrinsic_manifold_pts = manifold.intrinsic_to_extrinsic(intrinsic_manifold_pts)
        logging.info(f'Regenerated extrinsic Coordinates:\n{[m_pt.location for m_pt in extrinsic_manifold_pts]}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_extrinsic_to_spherical(self):
        intrinsic = False
        manifold = HypersphereSpace(True, intrinsic)
        # Create Manifold points with default extrinsic coordinates
        random_samples = manifold.sample(2)
        manifold_pts = [
            ManifoldPoint(f'id{index}', value, None, False, intrinsic)
            for index, value in enumerate(random_samples)
        ]
        logging.info(f'Original extrinsic coordinates:\n{[m_pt.location for m_pt in manifold_pts]}')

        intrinsic_manifold_pts = manifold.extrinsic_to_intrinsic(manifold_pts)
        logging.info(f'Intrinsic Coordinates:\n{[m_pt.location for m_pt in intrinsic_manifold_pts]}')

        spherical_manifold_pts = manifold.extrinsic_to_spherical(manifold_pts)
        logging.info(f'Spherical Coordinates:\n{[m_pt.location for m_pt in spherical_manifold_pts]}')

    def test_extrinsic_to_polar(self):
        intrinsic = False
        manifold = HypersphereSpace(True, intrinsic)
        # Create Manifold points with default extrinsic coordinates
        random_samples = manifold.sample(2)
        manifold_pts = [
            ManifoldPoint(f'id{index}', value, None, False, intrinsic)
            for index, value in enumerate(random_samples)
        ]

        polar_coordinates = manifold.extrinsic_to_intrinsic_polar(manifold_pts)
        logging.info(polar_coordinates)


if __name__ == '__main__':
    unittest.main()