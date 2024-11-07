import unittest
import numpy as np
from manifolds.manifoldpoint import ManifoldPoint
from manifolds.hyperspherespace import HypersphereSpace


class TestManifoldPoint(unittest.TestCase):

    def test_to_intrinsic(self):
        hypersphere_space = HypersphereSpace(True)
        manifold_pt = TestManifoldPoint.__create_manifold_point(hypersphere_space)
        intrinsic_coordinates = manifold_pt.to_intrinsic(hypersphere_space.space)
        assert len(intrinsic_coordinates) == 2
        print(f'To intrinsic: {intrinsic_coordinates}')

    def test_to_extrinsic(self):
        intrinsic = True
        hypersphere_space = HypersphereSpace(True, intrinsic)
        manifold_pt = TestManifoldPoint.__create_manifold_point(hypersphere_space)
        extrinsic_coordinates = manifold_pt.to_extrinsic(hypersphere_space.space)
        assert len(extrinsic_coordinates) == 3
        print(f'To extrinsic: {extrinsic_coordinates}')

    def test_to_intrinsic_polar(self):
        hypersphere_space = HypersphereSpace(True)
        manifold_pt = TestManifoldPoint.__create_manifold_point(hypersphere_space)
        polar_coordinates = manifold_pt.to_intrinsic_polar(hypersphere_space.space)
        assert len(polar_coordinates) == 2
        print(f'To polar: {polar_coordinates}')

    @staticmethod
    def __create_manifold_point(hypersphere_space: HypersphereSpace) -> ManifoldPoint:
        # Create a Manifold point with default extrinsic coordinates
        is_intrinsic = hypersphere_space.space.default_coords_type == 'intrinsic'
        random_sample: np.array = hypersphere_space.sample(1) if not is_intrinsic \
            else np.array([0.3, 0.5])

        return ManifoldPoint(
            id='id1',
            location=random_sample,
            tgt_vector=None,
            geodesic=False,
            intrinsic=is_intrinsic
        )


if __name__ == '__main__':
    unittest.main()