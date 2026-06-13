from unittest import TestCase

import unittest
import numpy as np
import logging
import os
import python
from python import SKIP_REASON


class TestNumpy(TestCase):
    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_slice(self):
        try:
            x = TestNumpy.__generate_3d_array(32, 4)
            logging.info(f'\nx:------\n{x}')
            logging.info(f'\nx[::1]:------\n{x[::1]}')
            logging.info(f'\nx[::2]:------\n{x[::2]}')
            logging.info(f'\nx[::3]:------\n{x[::3]}')
            logging.info(f'\nx[:1:]:------\n{x[:1:]}')
            logging.info(f'\nx[:2:]:------\n{x[:2:]}')
            logging.info(f'\nx[0::]:------\n{x[0::]}')
            logging.info(f'\nx[1::]:------\n{x[1::]}')
            logging.info(f'\nx[2::]:------\n{x[2::]}')
            logging.info(f'\nx[2::1]:------\n{x[2::1]}')
            logging.info(f'\nx[1:2,:,1]:------\n{x[1:2,:,1]}')
            logging.info(f'\nx[1:2:,1]:------\n{x[1:2:,1]}')
            logging.info(f'\nx[::,1]:------\n{x[::,1]}')
            logging.info(f'\nx[:2,:,1]:------\n{x[:2,:,1]}')
            logging.info(f'\nx[0,0,0]:------\n{x[0,0,0]}')
            logging.info(f'\nx[:0:,1]:------\n{x[:0:,1]}')
            logging.info(f'\nx[0:,:,1]:------\n{x[0:,:,1]}')
            logging.info(f'\nx[2::,0]:------\n{x[2::,0]}')
            logging.info(f'\nx[1:3,0:2,1]:------\n{x[1:3,0:2,1]}')
            logging.info(f'\nx[1:3,:,1]:------\n{x[1:3,:,1]}')
            logging.info(f'\nx[1:3,:,0]:------\n{x[1:3,:,0]}')
            logging.info(f'\nx[1:2,:,0]:------\n{x[1:2,:,0]}')
            logging.info(f'\nx[:,0:,0]:------\n{x[:,0:,0]}')
        except Exception as e:
            self.fail(str(e))

    def test_flattening(self):
        x = np.array([111.0,112.0,121.0,122.0,131.0,132.0,211.0,212.0,221.0,222.0,231.0,232.0])
        y = x.reshape(2, 3, 2)
        logging.info(y)

    @staticmethod
    def __generate_3d_array(sz: int, width: int):
        x = np.arange(100, 100+sz, 1)
        if sz != width*width*2:
            raise Exception(f'Size {sz} and reshape {width} are incompatible')
        return x.reshape(width, width, 2)

