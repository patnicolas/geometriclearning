import unittest
from dataset.labeledloader import LabeledLoader
from dataset.unlabeleddataset import UnlabeledDataset
import torch
import numpy as np


class LabeledLoaderTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_load_tensor_transform(self):
        try:
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
            self.assertTrue(len(output) > 0)
        except FileNotFoundError as e:
            print(f'Error: {str(e)}')
            self.assertFalse(True)
        except Exception as e:
            print(f'Error: {str(e)}')
            self.assertFalse(True)

    @unittest.skip('Ignore')
    def test_load_csv_file(self):
        try:
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
            self.assertTrue(len(results) > 0)
        except FileNotFoundError as e:
            print(f'Error: {str(e)}')
            self.assertFalse(True)
        except Exception as e:
            print(f'Error: {str(e)}')
            self.assertFalse(True)

    def test_load_mnist(self):
        try:
            normalizing_mean = 0.2
            normalizing_std_dev = 0.35
            resize_factor = 32
            batch_size = 8
            train_eval_split_ratio = 0.85

            labeled_loader = LabeledLoader(batch_size, train_eval_split_ratio)
            train_loader, test_loader = labeled_loader.load_mnist([normalizing_mean, normalizing_std_dev], resize_factor)

            images, labels = next(iter(train_loader))
            import matplotlib.pyplot as plt
            for img in images:
                plt.imshow(img.squeeze())
                plt.show()

        except FileNotFoundError as e:
            print(f'Error: {str(e)}')
            self.assertFalse(True)
        except Exception as e:
            print(f'Error: {str(e)}')
            self.assertFalse(True)


if __name__ == '__main__':
    unittest.main()

