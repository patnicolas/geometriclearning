import unittest
import numpy as np
from topology.simplicial_model import SimplicialModel
import logging
import python

class SimplicialModelTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init(self):
        features = np.array([
            [0.4, 0.6, -0.8, 0.1],  # node 1
            [0.0, -0.3, 0.2, -0.5], # node 2
            [0.5, -0.3, 0.0, -0.1], # node 3
            [0.8, 0.1, -0.7,  0.0], # node 4
            [0.3, -0.4, -0.2, 1.0], # node 5
        ])
        edge_set = np.array([[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]])
        face_set = np.array([[2, 3, 4], [1, 2, 3]])
        try:
            simplicial_model = SimplicialModel(features, edge_set, face_set)
            logging.info(simplicial_model)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_show(self):
        features = np.array([
            [0.4, 0.6, -0.8, 0.1],      # node 1
            [0.0, -0.3, 0.2, -0.5],     # node 2
            [0.5, -0.3, 0.0, -0.1],     # node 3
            [0.8, 0.1, -0.7,  0.0],     # node 4
            [0.3, -0.4, -0.2, 1.0],     # node 5
            [0.0, -0.9, 0.0, 0.6],      # node 6
            [1.0, 0.5, -0.1, 0.0],      # node 7
        ])
        edge_set = np.array([[1, 2], [1, 3], [2, 3], [5, 6], [3, 4], [1, 6], [4, 5], [3, 6], [4, 6], [6, 7], [7, 1]])
        face_set = np.array([[4, 3, 6], [1, 2, 3], [5, 4, 6], [6, 7, 1]])
        try:
            simplicial_model = SimplicialModel(features, edge_set, face_set)
            simplicial_model.show()
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_adjacency(self):
        features = np.array([
            [0.4, 0.6, -0.8, 0.1],  # node 1
            [0.0, -0.3, 0.2, -0.5], # node 2
            [0.5, -0.3, 0.0, -0.1], # node 3
            [0.8, 0.1, -0.7,  0.0], # node 4
            [0.3, -0.4, -0.2, 1.0], # node 5
        ])
        edge_set = np.array([[1, 2],[1, 3], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]])
        face_set = np.array([[2, 3, 4], [1, 2, 3]])
        try:
            simplicial_model = SimplicialModel(features, edge_set, face_set)
            logging.info(f'Adjacency matrix:\n{simplicial_model.adjacency_matrix()}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_incidence_directed(self):
        features = np.array([
            [0.4, 0.6, -0.8, 0.1],  # node 1
            [0.0, -0.3, 0.2, -0.5], # node 2
            [0.5, -0.3, 0.0, -0.1], # node 3
            [0.8, 0.1, -0.7,  0.0], # node 4
            [0.3, -0.4, -0.2, 1.0], # node 5
        ])
        edge_set = np.array([[1, 2],[1, 3], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]])
        face_set = np.array([[2, 3, 4], [1, 2, 3]])
        try:
            simplicial_model = SimplicialModel(features, edge_set, face_set)
            for rank in range(0, 3):
                logging.info(f'\nDirected incidence matrix rank {rank}:\n{simplicial_model.incidence_matrix(rank=rank)}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_incidence_undirected(self):
        features = np.array([
            [0.4, 0.6, -0.8, 0.1],  # node 1
            [0.0, -0.3, 0.2, -0.5], # node 2
            [0.5, -0.3, 0.0, -0.1], # node 3
            [0.8, 0.1, -0.7,  0.0], # node 4
            [0.3, -0.4, -0.2, 1.0], # node 5
        ])
        edge_set = np.array([[1, 2],[1, 3], [2, 3], [2, 4], [3, 4], [2, 5], [4, 5]])
        face_set = np.array([[2, 3, 4], [1, 2, 3]])
        try:
            simplicial_model = SimplicialModel(features, edge_set, face_set)
            for rank in range(0, 3):
                undirected_incidence = simplicial_model.incidence_matrix(rank=rank, directed_graph=False)
                logging.info(f'\nUndirected incidence matrix rank {rank}:\n{undirected_incidence}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)



