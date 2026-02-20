
import unittest
from unittest import TestCase
import os
import numpy as np
import torch
import logging
import python
from python import SKIP_REASON


class TestPyTorch(TestCase):

    def test_chunking_1(self):
        x = 7
        y = "2"
        print(x+y)
        a = np.array([0.0, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        x = torch.from_numpy(a)
        chunks = torch.chunk(x, 3, dim=0)
        res = '\n'.join([str(ch) for ch in chunks])
        logging.info(f'\nChunked:\n{res}')

    def test_chunking_2(self):
        a = np.array([0.0, 1.0, 0.4, 0.0, 1.0, 1.0, 0.0, 0.9, 0.4, 0.3]).reshape((2, 5))
        x = torch.from_numpy(a)
        chunks = torch.chunk(x, 3, dim=0)
        res = '\n'.join([str(ch.float()) for ch in chunks])
        logging.info(f'Input:\n{str(a)}\nChunked:\n{res}')

        chunks = torch.chunk(x, 3, dim=1)
        res = '\n'.join([str(ch.float()) for ch in chunks])
        logging.info(f'Input:\n{str(a)}\nChunked:\n{res}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_sparse(self):
        a = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        x = torch.from_numpy(a).to_sparse()
        logging.info(x)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_argmax(self):
        x = torch.tensor([[0.4, 0.5, 0.6], [1.0, 0.2, 0.7]])
        max_all = torch.argmax(x)      # index = 3 for 1.0
        max_1 = torch.argmax(x, dim=0) # indices 1, 0, 1 for 1.0 0.5 and 0.7
        max_2 = torch.argmax(x, dim=1) # indices 2, 0 for [0.6 and 1.0
        logging.info(f'{max_all}  {max_1}  {max_2}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_tensor(self):
        x = TestPyTorch.__generate_3d_tensor(sz=32, width=4, torch_device='cpu')
        logging.info(f'\nx:------\n{x.shape}\n{x}')
        logging.info(f'\nx[::1]:------\n{x[::1]}')
        logging.info(f'\nx[:0:,1]:------\n{x[:0:,1]}')
        logging.info(f'\nx[0:,:,1]:------\n{x[0:,:,1]}')
        logging.info(f'\nx[2::,0]:------\n{x[2::,0]}')
        logging.info(f'\nx[1:3,0:2,1]:------\n{x[1:3,0:2,1]}')
        logging.info(f'\nx[1:3,:,1]:------\n{x[1:3,:,1]}')
        logging.info(f'\nx[1:3,:,0]:------\n{x[1:3,:,0]}')
        logging.info(f'\nx[1:2,:,0]:------\n{x[1:2,:,0]}')
        logging.info(f'\nx[:,0:,0]:------\n{x[:,0:,0]}')
        logging.info(f'\nunsqueeze(0):------\n{x.unsqueeze(0)}')
        logging.info(f'\nview(1,4,4,2):------\n{  x.view(1,4,4,2)}')
        logging.info(f'\nview(1,4,4,-1):------\n{x.view(1,4,4,-1)}')
        logging.info(f'\nunsqueeze(1):------\n{x.unsqueeze(1)}')
        logging.info(f'\nview(4,1,4,2):------\n{x.view(4,1,4,2)}')
        logging.info(f'\nunsqueeze(2).unsqueeze(2).shape:------\n{x.unsqueeze(2).unsqueeze(2).shape}')


    @staticmethod
    def __generate_3d_tensor(sz: int, width: int, torch_device):
        x = np.arange(100, 100 + sz, 1)
        if sz != width * width * 2:
            raise Exception(f'Size {sz} and reshape {width} are incompatible')
        return torch.tensor(x.reshape(width, width, 2), dtype=torch.float32, device=torch_device)