import unittest
from topology.cell.featured_cell import FeaturedCell
from toponetx.classes.cell import Cell
import numpy as np
import logging
import python

class FeaturedCellTest(unittest.TestCase):

    def test_init_1(self):
        features = FeaturedCell(Cell(elements=[1, 2], rank=1), features=np.array([1.5, 6.0]))
        logging.info(features)
        self.assertTrue(True)

    def test_init_2(self):
        try:
            features = FeaturedCell.build(indices=[1, 2], rank=1, features=np.array([1.5, 6.0]))
            logging.info(features)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(True)

    def test_init_3(self):
        try:
            features = FeaturedCell.build(indices=[1, 2], rank=0, features=np.array([1.5, 6.0]))
            logging.info(features)
            self.assertFalse(False)
        except ValueError as e:
            logging.error(e)
            self.assertFalse(False)

