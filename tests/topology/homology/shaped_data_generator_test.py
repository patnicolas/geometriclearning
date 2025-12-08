import unittest
import logging
import python

from topology.homology.shaped_data_generator import ShapedDataGenerator, ShapedDataDisplay

class ShapedDataGeneratorTest(unittest.TestCase):

    def test_show_swiss_roll(self):
        try:
            swiss_roll_data_display = ShapedDataDisplay(ShapedDataGenerator.SWISS_ROLL)
            swiss_roll_data_display.__call__({'n': 200}, noise=0.25)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    def test_show_sphere(self):
        try:
            sphere_data_display = ShapedDataDisplay(ShapedDataGenerator.SPHERE)
            sphere_data_display.__call__({'n': 200}, noise=0.25)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    def test_show_torus(self):
        try:
            torus_data_display = ShapedDataDisplay(ShapedDataGenerator.TORUS)
            torus_data_display.__call__({'n': 200}, noise=0.25)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)
