import unittest
from geometry.kmeans_on_manifold import KMeansOnManifold, KMeansCluster
from typing import List, AnyStr
import logging
import os
import python
from python import SKIP_REASON

class KMeansOnManifoldTest(unittest.TestCase):
    num_samples = 500
    num_clusters = 4

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init(self):
        num_samples = 100
        num_clusters = 3
        kmeans = KMeansOnManifold(num_samples, num_clusters)
        logging.info(str(kmeans))

    def test_euclidean(self):
        kmeans = KMeansOnManifold(KMeansOnManifoldTest.num_samples, KMeansOnManifoldTest.num_clusters)
        kmeans_cluster = kmeans.euclidean_clustering()
        logging.info(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters in Euclidean space --------\n{KMeansOnManifoldTest.__str(kmeans_cluster)}')

    def test_riemannian_von_mises_fisher(self):
        kmeans = KMeansOnManifold(
            KMeansOnManifoldTest.num_samples,
            KMeansOnManifoldTest.num_clusters,
            'random_von_mises_fisher')
        kmeans_cluster = kmeans.riemannian_clustering()
        logging.info(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters with von-mises-Fisher distribution --------\n{KMeansOnManifoldTest.__str(kmeans_cluster)}')


    def test_riemannian_random_normal(self):
        kmeans = KMeansOnManifold(
            KMeansOnManifoldTest.num_samples,
            KMeansOnManifoldTest.num_clusters,
            'random_riemann_normal')
        kmeans_cluster = kmeans.riemannian_clustering()
        logging.info(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters with normal distribution  --------\n{KMeansOnManifoldTest.__str(kmeans_cluster)}')

    def test_riemannian_random_uniform(self):
        kmeans = KMeansOnManifold(KMeansOnManifoldTest.num_samples, KMeansOnManifoldTest.num_clusters, 'random_uniform')
        kmeans_clusters = kmeans.riemannian_clustering()
        logging.info(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters with uniform distribution  --------\n{KMeansOnManifoldTest.__str(kmeans_clusters)}')

    def test_riemannian_constrained_random_uniform(self):
        kmeans = KMeansOnManifold(KMeansOnManifoldTest.num_samples, KMeansOnManifoldTest.num_clusters, 'constrained_random_uniform')
        kmeans_clusters = kmeans.riemannian_clustering()
        logging.info(f'\n{KMeansOnManifoldTest.num_samples} random samples on {KMeansOnManifoldTest.num_clusters} '
              f'clusters with constrained uniform distribution  --------\n{KMeansOnManifoldTest.__str(kmeans_clusters)}')

    def test_evaluate(self):
        kmeans = KMeansOnManifold(KMeansOnManifoldTest.num_samples, KMeansOnManifoldTest.num_clusters)
        logging.info(f'Evaluate: {kmeans.evaluate()}')

    @staticmethod
    def __str(cluster_result: List[KMeansCluster])-> AnyStr:
        return '\n'.join([f'Cluster #{idx + 1}\n{str(cluster_res)}' for idx, cluster_res in enumerate(cluster_result)])


if __name__ == '__main__':
    unittest.main()