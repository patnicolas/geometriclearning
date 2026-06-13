import unittest
import numpy as np
import logging
import python
from topology.hypergraph.featured_hyperedge import FeaturedHyperEdge


class FeaturedHyperEdgeTest(unittest.TestCase):

    def test_init(self):
        hyperedge = FeaturedHyperEdge.build(hyperedge_indices=(1, 3, 4, 5), rank=2, features=np.array([0.5, 0.8, 1.2]))
        indices = hyperedge.get_indices()
        logging.info(indices)
        self.assertTrue(True)