import unittest
from python.dataset.labeledloader import LabeledLoader
from python.dataset.labeleddataset import LabeledDataset
from python.dataset.unlabeleddataset import UnlabeledDataset
import torch
import numpy as np


class LabeledLoaderTest(unittest.TestCase):

    def test_load_tensor_transform(self):
        batch_size = 4
        train_eval_split_ratio = 0.85
        dataset_loader = LabeledLoader(batch_size, train_eval_split_ratio)
        features = torch.randn(128, 8)
        labels = torch.Tensor([1.0 if np.random.rand() > 0.5 else 0.0 for _ in range(128)])

        def logarithm_scale(x: torch.Tensor) -> torch.Tensor:
            return torch.log(x + 1.0)

        train_set, eval_set = dataset_loader.from_tensor_transform(features, labels, logarithm_scale)
        output = '\n'.join([str(train_data) for idx, train_data in enumerate(train_set) if idx < 4])
        print(output)

    def test_load_file(self):
        filename = '/users/patricknicolas/dev/geometriclearning/data/wages_cleaned.csv'
        df = UnlabeledDataset.data_frame(filename)
        df = df[['Reputation', 'Age', 'Caps', 'Apps', 'Salary']]
        average_salary = df['Salary'].mean()
        df['Top_player'] = np.where(df['Salary'] > average_salary, 1.0, 0.0)

        batch_size = 4
        train_eval_split_ratio = 0.85
        dataset_loader = LabeledLoader(batch_size, train_eval_split_ratio)
        train_loader, eval_loader = dataset_loader.from_dataframes(df, df['Top_player'])

        results = [(item[0].numpy(), item[1].numpy()) for idx, item in enumerate(train_loader) if idx < 3]
        print(f'First data features:\n{results[0][0]})')
        print(f'First data labels:\n{results[0][1]})')
        print(f'First 3 values:\n{results})')


if __name__ == '__main__':
    unittest.main()

