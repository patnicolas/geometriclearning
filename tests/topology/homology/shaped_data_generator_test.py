import unittest
import logging
import python
from topology.homology.shaped_data_generator import ShapedDataGenerator, ShapedDataDisplay

class ShapedDataGeneratorTest(unittest.TestCase):

    @unittest.skip("skip_test")
    def test_show_uniform_random(self):
        try:
            uniform_data_display = ShapedDataDisplay(ShapedDataGenerator.UNIFORM)
            uniform_data_display.__call__({'n': 200})
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip("skip_test")
    def test_show_swiss_roll(self):
        try:
            swiss_roll_data_display = ShapedDataDisplay(ShapedDataGenerator.SWISS_ROLL)
            swiss_roll_data_display.__call__({'n': 200}, noise=0.25)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip("skip_test")
    def test_show_circle(self):
        try:
            sphere_data_display = ShapedDataDisplay(ShapedDataGenerator.CIRCLE)
            sphere_data_display.__call__({'n': 2000}, noise=0.05)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    def test_data_for_circle(self):
        try:
            sphere_data_display = ShapedDataDisplay(ShapedDataGenerator.CIRCLE)
            raw_data, shaped_data, _ = sphere_data_display.get_data({'n': 200}, noise=0.05, limit=20)
            logging.info(f'\nRaw data:\n{raw_data}\nShaped data:\n{shaped_data}')
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip("skip_test")
    def test_show_sphere(self):
        try:
            sphere_data_display = ShapedDataDisplay(ShapedDataGenerator.SPHERE)
            sphere_data_display.__call__({'n': 20}, noise=0.25)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    @unittest.skip("skip_test")
    def test_show_torus(self):
        try:
            torus_data_display = ShapedDataDisplay(ShapedDataGenerator.TORUS)
            torus_data_display.__call__({'n': 200}, noise=0.25)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)
