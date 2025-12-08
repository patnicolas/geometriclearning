import unittest

import logging
import python
from topology.homology.persistence_diagrams import PersistenceDiagrams
from topology.homology.shaped_data_generator import ShapedDataGenerator


class PersistenceDiagramsTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_display_sphere(self):
        try:
            num_raw_points = 256
            persistence_diagrams = PersistenceDiagrams.build(props={'n': num_raw_points}, 
                                                             shaped_data_generator=ShapedDataGenerator.SPHERE)
            persistence_diagrams.display()
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignore')
    def test_display_swiss_roll(self):
        try:
            persistence_diagrams = PersistenceDiagrams.build(props={'n': 384, 'noise': 0.60},
                                                             shaped_data_generator=ShapedDataGenerator.SWISS_ROLL)
            persistence_diagrams.display()
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignore')
    def test_display_torus(self):
        try:
            persistence_diagrams = PersistenceDiagrams.build(props={'n': 256, 'c': 1, 'a': 0.7, 'noise': 0.2},
                                                             shaped_data_generator=ShapedDataGenerator.TORUS)
            persistence_diagrams.display()
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignore')
    def test_display_random_normal(self):
        try:
            persistence_diagrams = PersistenceDiagrams.build(props={'n': 256},
                                                             shaped_data_generator=ShapedDataGenerator.NORMAL)
            persistence_diagrams.display()
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    def test_display_random_uniform(self):
        try:
            persistence_diagrams = PersistenceDiagrams.build(props={'n': 2048},
                                                             shaped_data_generator=ShapedDataGenerator.UNIFORM)
            persistence_diagrams.display()
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)
