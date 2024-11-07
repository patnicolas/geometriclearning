import unittest
import torch
from dataset.unlabeled_dataset import UnlabeledDataset
import numpy as np


class UnlabeledDatasetTest(unittest.TestCase):

    def test_init(self):
        data = torch.randn(128, 8)

        def logarithm_scale(x: torch.Tensor) -> torch.Tensor:
            return torch.log(x+1)

        tensor_dataset = UnlabeledDataset(data, logarithm_scale)
        print(f'First item: {tensor_dataset[0]}, Second Item: {tensor_dataset[1]}')

    def test_from_numpy(self):
        data = np.linspace(0.0, 10.0, 100)

        def logarithm_scale(x: torch.Tensor) -> torch.Tensor:
            return torch.log(x+1.0)

        tensor_dataset = UnlabeledDataset.from_numpy(data, logarithm_scale)
        output = ' '.join([str(float(el)) for el in tensor_dataset])
        print(output)

    def test_from_file(self):
        filename = '/users/patricknicolas/dev/geometriclearning/data/wages_cleaned.csv'
        tensor_dataset = UnlabeledDataset.from_file(filename, ['Reputation', 'Age', 'Caps', 'Salary'])
        print(f'Tensor from file:\n{repr(tensor_dataset)}')


if __name__ == '__main__':
    unittest.main()
