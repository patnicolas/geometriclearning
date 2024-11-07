import unittest
import torch
from dataset.tdataset import TDataset
from dataset.dataset_exception import DatasetException


class TDatasetTest(unittest.TestCase):

    def test_torch_to_df(self):
        filename = '/users/patricknicolas/dev/geometriclearning/data/MNIST/processed/training.pt'

        try:
            x = torch.tensor([
                [[0.3, 0.6],
                 [0.5, 1.1],
                 [1.2, 4.3]],
                [[3.3, 3.6],
                 [-3.5, 5.1],
                 [11.2, -4.3]]]
            )

            self.assertTrue(type(x) == torch.Tensor)
            df = TDataset.torch_to_df(x)
            print(df[0])
        except DatasetException as e:
            print(f'Error: {str(e)}')
            self.assertFalse(True)

    def test_torch_to_df(self):
        filename = '/users/patricknicolas/dev/geometriclearning/data/MNIST/processed/training.pt'
        try:
            dfs = TDataset.torch_to_dfs(filename)
            print(dfs[0])
        except DatasetException as e:
            print(f'Error: {str(e)}')
            self.assertFalse(True)


if __name__ == '__main__':
    unittest.main()