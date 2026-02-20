import unittest

import logging
import python
from topology.homology.persistence_diagrams import PersistenceDiagrams
from topology.homology.shaped_data_generator import ShapedDataGenerator


class PersistenceDiagramsTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_display_sphere(self):
        try:
            num_raw_points = 512
            persistence_diagrams = PersistenceDiagrams.build(props={'n': num_raw_points, 'noise': 0.0},
                                                             shaped_data_generator=ShapedDataGenerator.SPHERE)
            persistence_diagrams.display()
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignore')
    def test_display_swiss_roll(self):
        try:
            persistence_diagrams = PersistenceDiagrams.build(props={'n': 384, 'noise': 0.2},
                                                             shaped_data_generator=ShapedDataGenerator.SWISS_ROLL)
            persistence_diagrams.display()
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignore')
    def test_display_torus(self):
        try:
            persistence_diagrams = PersistenceDiagrams.build(props={'n': 256, 'c': 20, 'a': 15, 'noise': 0.65},
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

    @unittest.skip('Ignore')
    def test_display_random_uniform(self):
        try:
            persistence_diagrams = PersistenceDiagrams.build(props={'n': 256},
                                                             shaped_data_generator=ShapedDataGenerator.UNIFORM)
            persistence_diagrams.display()
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)
