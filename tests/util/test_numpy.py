from unittest import TestCase

import unittest
import numpy as np
import constants


class TestNumpy(TestCase):
    @unittest.skip("Not needed")
    def test_slice(self):
        try:
            x = TestNumpy.__generate_3d_array(32, 4)
            constants.log_info(f'\nx:------\n{x}')
            constants.log_info(f'\nx[::1]:------\n{x[::1]}')
            constants.log_info(f'\nx[::2]:------\n{x[::2]}')
            constants.log_info(f'\nx[::3]:------\n{x[::3]}')
            constants.log_info(f'\nx[:1:]:------\n{x[:1:]}')
            constants.log_info(f'\nx[:2:]:------\n{x[:2:]}')
            constants.log_info(f'\nx[0::]:------\n{x[0::]}')
            constants.log_info(f'\nx[1::]:------\n{x[1::]}')
            constants.log_info(f'\nx[2::]:------\n{x[2::]}')
            constants.log_info(f'\nx[2::1]:------\n{x[2::1]}')
            constants.log_info(f'\nx[1:2,:,1]:------\n{x[1:2,:,1]}')
            constants.log_info(f'\nx[1:2:,1]:------\n{x[1:2:,1]}')
            constants.log_info(f'\nx[::,1]:------\n{x[::,1]}')
            constants.log_info(f'\nx[:2,:,1]:------\n{x[:2,:,1]}')
            constants.log_info(f'\nx[0,0,0]:------\n{x[0,0,0]}')
            constants.log_info(f'\nx[:0:,1]:------\n{x[:0:,1]}')
            constants.log_info(f'\nx[0:,:,1]:------\n{x[0:,:,1]}')
            constants.log_info(f'\nx[2::,0]:------\n{x[2::,0]}')
            constants.log_info(f'\nx[1:3,0:2,1]:------\n{x[1:3,0:2,1]}')
            constants.log_info(f'\nx[1:3,:,1]:------\n{x[1:3,:,1]}')
            constants.log_info(f'\nx[1:3,:,0]:------\n{x[1:3,:,0]}')
            constants.log_info(f'\nx[1:2,:,0]:------\n{x[1:2,:,0]}')
            constants.log_info(f'\nx[:,0:,0]:------\n{x[:,0:,0]}')
        except Exception as e:
            self.fail(str(e))

    def test_flattening(self):
        x = np.array([111.0,112.0,121.0,122.0,131.0,132.0,211.0,212.0,221.0,222.0,231.0,232.0])
        y = x.reshape(2, 3, 2)
        print(y)


    @staticmethod
    def __generate_3d_array(sz: int, width: int):
        x = np.arange(100, 100+sz, 1)
        if sz != width*width*2:
            raise Exception(f'Size {sz} and reshape {width} are incompatible')
        return x.reshape(width, width, 2)

