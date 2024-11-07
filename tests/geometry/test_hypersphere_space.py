import unittest
from manifolds.hyperspherespace import HypersphereSpace
from manifolds.manifoldpoint import ManifoldPoint
from manifolds. spacevisualization import VisualizationParams, SpaceVisualization


class TestGeometricSpace(unittest.TestCase):

    @unittest.skip('ignore')
    def test_sample_hypersphere(self):
        num_samples = 180
        manifold = HypersphereSpace()
        data = manifold.sample(num_samples)
        print(f'Hypersphere:\n{str(data)}')

    @unittest.skip('ignore')
    def test_hypersphere(self):
        num_samples = 8
        style = {'color': 'red', 'linestyle': '--', 'label': 'Edges'}
        manifold = HypersphereSpace()
        print(str(manifold))
        data = manifold.sample(num_samples)
        visualParams = VisualizationParams("Data on Hypersphere", "locations", (8, 8), style, "3d")
        HypersphereSpace.show(visualParams, data)

    @unittest.skip('ignore')
    def test_tangent_vector(self):
        from geometricspace import GeometricSpace

        filename = '../../../data/hypersphere_data_1.txt'
        data = GeometricSpace.load_csv(filename)
        manifold = HypersphereSpace(True)
        samples = manifold.sample(3)
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
            print(f'Tangent vector: {vec} End point: {end_point}')
        manifold.show_manifold(manifold_points)

    @unittest.skip('ignore')
    def test_show_tangent_vector_geodesics(self):
        from geometricspace import GeometricSpace

        manifold = HypersphereSpace(True)
        samples = manifold.sample(2)
        manifold_points = [
            ManifoldPoint(
                id=f'data{index}',
                location=sample,
                tgt_vector=[0.5, 0.3, 0.5],
                geodesic=True) for index, sample in enumerate(samples)
        ]
        manifold.show_manifold(manifold_points)

    @unittest.skip('ignore')
    def test_euclidean_mean(self):
        manifold = HypersphereSpace(True)
        samples = manifold.sample(3)
        manifold_points = [
            ManifoldPoint(
                id=f'data{index}',
                location=sample) for index, sample in enumerate(samples)
        ]
        mean = manifold.euclidean_mean(manifold_points)
        print(mean)

    @unittest.skip('ignore')
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
        frechet_mean = manifold.frechet_mean(manifold_points[0], manifold_points[1])
        print(f'Euclidean mean: {euclidean_mean}\nFrechet mean: {frechet_mean}')
        assert manifold.belongs(frechet_mean)

        frechet_pt = ManifoldPoint(
            id='Frechet mean',
            location=frechet_mean,
            tgt_vector=[0.0, 0.0, 0.0],
            geodesic=False)
        manifold_points.append(frechet_pt)
        manifold.show_manifold(manifold_points, [euclidean_mean])

    @unittest.skip('ignore')
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
        print(f'From extrinsic Coordinates:\n{[m_pt.location for m_pt in manifold_pts]}')
        intrinsic_manifold_pts = manifold.extrinsic_to_intrinsic(manifold_pts)
        print(f'To intrinsic Coordinates: {[m_pt.location for m_pt in intrinsic_manifold_pts]}')

    @unittest.skip('ignore')
    def test_intrinsic_to_extrinsic(self):
        intrinsic = True
        manifold = HypersphereSpace(True, intrinsic)
        # Create Manifold points with default extrinsic coordinates
        random_samples = manifold.sample(2)
        manifold_pts = [
            ManifoldPoint(f'id{index}', value, None, False, intrinsic) for index, value in enumerate(random_samples)
        ]
        print(f'From intrinsic Coordinates:\n{[m_pt.location for m_pt in manifold_pts]}')
        extrinsic_manifold_pts = manifold.intrinsic_to_extrinsic(manifold_pts)
        print(f'To extrinsic Coordinates:\n{[m_pt.location for m_pt in extrinsic_manifold_pts]}')

    @unittest.skip('ignore')
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
        print(f'Original extrinsic coordinates:\n{[m_pt.location for m_pt in manifold_pts]}')
        intrinsic_manifold_pts = manifold.extrinsic_to_intrinsic(manifold_pts)
        print(f'Intrinsic Coordinates:\n{[m_pt.location for m_pt in intrinsic_manifold_pts]}')
        extrinsic_manifold_pts = manifold.intrinsic_to_extrinsic(intrinsic_manifold_pts)
        print(f'Regenerated extrinsic Coordinates:\n{[m_pt.location for m_pt in extrinsic_manifold_pts]}')

    @unittest.skip('ignore')
    def test_extrinsic_to_spherical(self):
        intrinsic = False
        manifold = HypersphereSpace(True, intrinsic)
        # Create Manifold points with default extrinsic coordinates
        random_samples = manifold.sample(2)
        manifold_pts = [
            ManifoldPoint(f'id{index}', value, None, False, intrinsic)
            for index, value in enumerate(random_samples)
        ]
        print(f'Original extrinsic coordinates:\n{[m_pt.location for m_pt in manifold_pts]}')

        intrinsic_manifold_pts = manifold.extrinsic_to_intrinsic(manifold_pts)
        print(f'Intrinsic Coordinates:\n{[m_pt.location for m_pt in intrinsic_manifold_pts]}')

        spherical_manifold_pts = manifold.extrinsic_to_spherical(manifold_pts)
        print(f'Spherical Coordinates:\n{[m_pt.location for m_pt in spherical_manifold_pts]}')

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
        print(polar_coordinates)


if __name__ == '__main__':
    unittest.main()