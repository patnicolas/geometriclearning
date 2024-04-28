from unittest import TestCase
import numpy as np
import torch
import constants


class TestPyTorch(TestCase):
    def test_sparse(self):
        import numpy
        a = numpy.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        x = torch.from_numpy(a).to_sparse()
        print(x)
        x


    def test_argmax(self):
        x = torch.tensor([[0.4, 0.5, 0.6], [1.0, 0.2, 0.7]])
        max_all = torch.argmax(x)      # index = 3 for 1.0
        max_1 = torch.argmax(x, dim=0) # indices 1, 0, 1 for 1.0 0.5 and 0.7
        max_2 = torch.argmax(x, dim=1) # indices 2, 0 for [0.6 and 1.0
        print(f'{max_all}  {max_1}  {max_2}')

    def test_tensor(self):
        x = TestPyTorch.__generate_3d_tensor(32, 4)
        print(f'\nx:------\n{x.shape}\n{x}')
        print(f'\nx[::1]:------\n{x[::1]}')
        print(f'\nx[:0:,1]:------\n{x[:0:,1]}')
        print(f'\nx[0:,:,1]:------\n{x[0:,:,1]}')
        print(f'\nx[2::,0]:------\n{x[2::,0]}')
        print(f'\nx[1:3,0:2,1]:------\n{x[1:3,0:2,1]}')
        print(f'\nx[1:3,:,1]:------\n{x[1:3,:,1]}')
        print(f'\nx[1:3,:,0]:------\n{x[1:3,:,0]}')
        print(f'\nx[1:2,:,0]:------\n{x[1:2,:,0]}')
        print(f'\nx[:,0:,0]:------\n{x[:,0:,0]}')
        print(f'\nunsqueeze(0):------\n{x.unsqueeze(0)}')
        print(f'\nview(1,4,4,2):------\n{  x.view(1,4,4,2)}')
        print(f'\nview(1,4,4,-1):------\n{x.view(1,4,4,-1)}')
        print(f'\nunsqueeze(1):------\n{x.unsqueeze(1)}')
        print(f'\nview(4,1,4,2):------\n{x.view(4,1,4,2)}')
        print(f'\nunsqueeze(2).unsqueeze(2).shape:------\n{x.unsqueeze(2).unsqueeze(2).shape}')

    @staticmethod
    def __generate_3d_tensor(sz: int, width: int):
        x = np.arange(100, 100 + sz, 1)
        if sz != width * width * 2:
            raise Exception(f'Size {sz} and reshape {width} are incompatible')
        return torch.tensor(x.reshape(width, width, 2), dtype=torch.float32, device=constants.torch_device)