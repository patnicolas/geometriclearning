import unittest
from dataset.unlabeled_loader import UnlabeledLoader
from dataset.unlabeled_dataset import UnlabeledDataset
import torch
import numpy as np


class UnlabeledLoaderTest(unittest.TestCase):

    def test_load_tensor_transform(self):
        batch_size = 4
        train_eval_split_ratio = 0.85
        dataset_loader = UnlabeledLoader(batch_size, train_eval_split_ratio)
        data = torch.randn(128, 8)

        def logarithm_scale(x: torch.Tensor) -> torch.Tensor:
            return torch.log(x + 1.0)
        train_set, eval_set = dataset_loader.from_tensor_transform(data, logarithm_scale)
        output = '\n'.join([str(train_data) for idx, train_data in enumerate(train_set) if idx < 4])
        print(output)

    def test_load_file(self):
        filename = '/users/patricknicolas/dev/geometric_learning/data/wages_cleaned.csv'
        dataset = UnlabeledDataset.from_file(filename, ['Reputation', 'Salary'])
        batch_size = 4
        train_eval_split_ratio = 0.85
        dataset_loader = UnlabeledLoader(batch_size, train_eval_split_ratio)
        train_loader, eval_loader = dataset_loader.from_dataset(dataset)
        results: np.array = [item.numpy() for idx, item in enumerate(train_loader) if idx < 3]
        print(f'First value:\n{results[0]})')
        print(f'First 3 values:\n{results})')


if __name__ == '__main__':
    unittest.main()