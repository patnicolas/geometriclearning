import unittest
import torch
from geometry.discrete.floyd_warshall import FloydWarshall

import logging
import python

class FloydWarshallTest(unittest.TestCase):

    def test_init(self):
        edge_index = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (4, 0), (1, 4), (2, 5), (4, 5)]
        weights = torch.Tensor([
            0.1, 0.4, 0.2, 0.1, 0.5, 0.7, 1.0, 0.6, 0.3
        ])

        floyd_warshall = FloydWarshall(edge_index, weights)
        logging.info(floyd_warshall)


    def test_call(self):
        edge_index = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (4, 0), (1, 4), (2, 5), (4, 5)]
        weights = torch.Tensor([
            0.1, 0.4, 0.2, 0.1, 0.5, 0.7, 1.0, 0.6, 0.3
        ])
        floyd_warshall = FloydWarshall(edge_index, weights)
        floyd_warshall()
        logging.info(floyd_warshall)

    def test_call_2(self):
        edge_index = [(0, 1), (0, 3), (1, 2), (2, 3)]
        weights = torch.Tensor([
            0.0, 5.0, 3.0, 1.0
        ])

        floyd_warshall = FloydWarshall(edge_index, weights)
        floyd_warshall()
        logging.info(floyd_warshall)
