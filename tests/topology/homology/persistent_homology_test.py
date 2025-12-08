import unittest
import logging
import python
from topology.homology.persistent_homology import PersistentHomology, ShapedDataGenerator

class PersistentHomologyTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_create_data_torus(self):
        try:
            num_raw_points = 260
            persistent_homology = PersistentHomology(ShapedDataGenerator.TORUS)
            data = persistent_homology.create_data({'n': num_raw_points})
            logging.info(data)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignore')
    def test_create_data_circle(self):
        try:
            num_raw_points = 260
            persistent_homology = PersistentHomology(ShapedDataGenerator.CIRCLE)
            data = persistent_homology.create_data({'n': num_raw_points})
            logging.info(data)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignore')
    def test_create_data_circle_failed(self):
        try:
            num_raw_points = 260
            persistent_homology = PersistentHomology(ShapedDataGenerator.CIRCLE)
            data = persistent_homology.create_data({'n': num_raw_points, 'noise': 2.7})
            logging.info(data)
            self.assertTrue(False)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(False)

    # @unittest.skip('Ignore')
    def test_create_plot_sphere(self):
        try:
            num_raw_points = 120
            persistent_homology = PersistentHomology(ShapedDataGenerator.SPHERE)
            persistent_homology.plot({'n': num_raw_points})
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    # @unittest.skip('Ignore')
    def test_create_plot_torus(self):
        try:
            num_raw_points = 260
            persistent_homology = PersistentHomology(ShapedDataGenerator.TORUS)
            persistent_homology.plot({'n': num_raw_points})
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    # @unittest.skip('Ignore')
    def test_create_plot_swiss_roll(self):
        try:
            num_raw_points = 260
            persistent_homology = PersistentHomology(ShapedDataGenerator.SWISS_ROLL)
            persistent_homology.plot({'n': num_raw_points})
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignore')
    def test_persistence_diagrams_sphere(self):
        try:
            num_raw_points = 1028
            persistent_homology = PersistentHomology(ShapedDataGenerator.SWISS_ROLL)
            persistent_homology.persistence_diagram(props={'n': num_raw_points})
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip('Ignore')
    def test_persistence_diagrams_noisy_sphere(self):
        try:
            num_raw_points = 128
            persistent_homology = PersistentHomology(ShapedDataGenerator.SWISS_ROLL)
            persistent_homology.persistence_diagram(props={'n': num_raw_points, 'noise': 0.0})
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)
