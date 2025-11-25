import unittest
import logging
from topology.hypergraph.featured_hyperedge import FeaturedHyperEdge
from topology.complex_laplacian import ComplexLaplacian
from topology import LaplacianType
import numpy as np
import python

from topology.hypergraph.featured_hypergraph import FeaturedHyperGraph


class FeaturedHypergraphTest(unittest.TestCase):

    def test_init_1(self):
        hyperedge1 = FeaturedHyperEdge.build(hyperedge_indices=(1, 3, 4, 5), rank=2, features=np.array([0.5, 0.8, 0.2]))
        hyperedge2 = FeaturedHyperEdge.build(hyperedge_indices=(2, 3, 4), rank=1, features=np.array([0.1, 0.0, 0.2]))
        hyperedge3 = FeaturedHyperEdge.build(hyperedge_indices=(1, 2, 4, 5), rank=2, features=np.array([0.0, 0.5, 0.1]))
        hyperedge4 = FeaturedHyperEdge.build(hyperedge_indices=(1, 3, 4), rank=2, features=np.array([0.9, 0.1, 0.7]))
        hyperedge5 = FeaturedHyperEdge.build(hyperedge_indices=(4, 5), rank=1, features=np.array([0.2, 0.6, 0.2]))

        featured_hypergraph = FeaturedHyperGraph(
            featured_hyperedges=[hyperedge1, hyperedge2, hyperedge3, hyperedge4, hyperedge5],
        )
        logging.info(featured_hypergraph)

    def test_init_2(self):
        try:
            hyperedge_indices_list = [(1, 3, 4, 5), (2, 3, 4), (1, 2, 4, 5), (1, 3, 4), (4, 5)]
            rank_list = [2, 2, 2, 2, 1]
            features_list = [[0.5, 0.8, 0.2], [0.1, 0.0, 0.2], [0.0, 0.5, 0.1], [0.9, 0.1, 0.7], [0.2, 0.6, 0.2]]
            featured_hypergraph = FeaturedHyperGraph.build(
                hyperedge_indices_list=hyperedge_indices_list,
                ranks=rank_list,
                features_list=features_list
            )
            logging.info(featured_hypergraph)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    def test_adjacency(self):
        try:
            hyperedge_indices_list = [(1, 2), (1, 4), (1, 2, 4, 3), (3, 5), (5, 6, 4), (1, 2, 4, 3, 5, 6)]
            rank_list = [1, 1, 2, 1, 2, 3]
            features_list = [[0.5, 0.8], [0.1, 0.0], [0.0, 0.5], [0.9, 0.1], [0.2, 0.5], [1.0, 0.2]]
            featured_hypergraph = FeaturedHyperGraph.build(
                hyperedge_indices_list=hyperedge_indices_list,
                ranks=rank_list,
                features_list=features_list
            )
            adjacency_matrix = featured_hypergraph.adjacency_matrix(ranks=(0, 1))
            logging.info(f'\nAdjacency Matrix (0, 1):\n{adjacency_matrix}')

            adjacency_matrix = featured_hypergraph.adjacency_matrix(ranks=(0, 2))
            logging.info(f'\nAdjacency Matrix (0, 2):\n{adjacency_matrix}')

            adjacency_matrix = featured_hypergraph.adjacency_matrix(ranks=(1, 2))
            logging.info(f'\nAdjacency Matrix (1, 2):\n{adjacency_matrix}')

            adjacency_matrix = featured_hypergraph.adjacency_matrix(ranks=(1, 3))
            logging.info(f'\nAdjacency Matrix (1, 3):\n{adjacency_matrix}')
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    def test_incidence(self):
        try:
            hyperedge_indices_list = [(1, 2), (1, 4), (1, 2, 4, 3), (3, 5), (5, 6, 4), (1, 2, 4, 3, 5, 6)]
            rank_list = [1, 1, 2, 1, 2, 3]
            features_list = [[0.5, 0.8], [0.1, 0.0], [0.0, 0.5], [0.9, 0.1], [0.2, 0.5], [1.0, 0.2]]
            featured_hypergraph = FeaturedHyperGraph.build(
                hyperedge_indices_list=hyperedge_indices_list,
                ranks=rank_list,
                features_list=features_list
            )
            incidence_matrix = featured_hypergraph.incidence_matrix(ranks=(0, 1))
            logging.info(f'\nIncidence Matrix (0, 1):\n{incidence_matrix}')

            incidence_matrix = featured_hypergraph.adjacency_matrix(ranks=(0, 2))
            logging.info(f'\nincidence Matrix (0, 2):\n{incidence_matrix}')

            incidence_matrix = featured_hypergraph.adjacency_matrix(ranks=(1, 2))
            logging.info(f'\nincidence Matrix (1, 2):\n{incidence_matrix}')
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    def test_to_simplicial_complex(self):
        hyperedge_indices_list = [(1, 2), (1, 4), (1, 2, 4, 3), (3, 5), (5, 6, 4), (1, 2, 4, 3, 5, 6)]
        rank_list = [1, 1, 2, 1, 2, 3]
        features_list = [[0.5, 0.8], [0.1, 0.0], [0.0, 0.5], [0.9, 0.1], [0.2, 0.5], [1.0, 0.2]]
        featured_hypergraph = FeaturedHyperGraph.build(
            hyperedge_indices_list=hyperedge_indices_list,
            ranks=rank_list,
            features_list=features_list
        )
        featured_hypergraph.set_simplicial_complex()
        logging.info(featured_hypergraph)
        complex_laplacian = ComplexLaplacian(
                laplacian_type=LaplacianType.HodgeLaplacian,
                rank=2,
                signed=False
            )
        laplacian = featured_hypergraph.laplacian(complex_laplacian)
        logging.info(f'\nHodge Laplacian rank 2:\n{laplacian}')

