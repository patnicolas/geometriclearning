import unittest
from dl.block.mlp_block import MLPBlock
from dl.model.mlp_model import MLPModel
from dl.model.vae_model import VAEModel
from dl.block import VAEException, MLPException
from dl.training import TrainingException
from metric.metric_type import MetricType
from dataset.tensor.unlabeled_loader import UnlabeledLoader
from dl.training.exec_config import ExecConfig
from dl.training.vae_training import VAETraining
from dl.training.hyper_params import HyperParams
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import AnyStr
import torch.nn as nn
import logging
import python


class VAETest(unittest.TestCase):

    def test_init(self):
        features = ['age', 'sex', 'chest pain type', 'cholesterol', 'fasting blood sugar', 'max heart rate',
                    'exercise angina', 'ST slope']
        try:
            hidden_block = MLPBlock(block_id='hidden',
                                    layer_module=nn.Linear(in_features=len(features), out_features=4),
                                    activation_module=nn.ReLU())
            output_block = MLPBlock(block_id='latent',
                                    layer_module=nn.Linear(in_features=4, out_features=4),
                                    activation_module=nn.ReLU())
            vae_model = VAEModel(model_id='Autoencoder',
                                 encoder=MLPModel(model_id='encoder', mlp_blocks=[hidden_block, output_block]),
                                 latent_dim=6)
            logging.info(vae_model)
            self.assertTrue(True)
        except (AssertionError | VAEException | MLPException| TrainingException) as e:
            logging.error(e)
            self.assertTrue(False)


    def test_train_1(self):
        from python.metric.metric import Metric
        from python.metric.built_in_metric import BuiltInMetric

        try:
            features = ['age', 'sex', 'chest pain type', 'cholesterol', 'fasting blood sugar', 'max heart rate',
                        'exercise angina', 'ST slope']
            hidden_block = MLPBlock(block_id='hidden',
                                    layer_module=nn.Linear(in_features=len(features), out_features=4),
                                    activation_module=nn.ReLU())
            output_block = MLPBlock(block_id='latent',
                                    layer_module=nn.Linear(in_features=4, out_features=4),
                                    activation_module=nn.ReLU())
            vae_model = VAEModel(model_id='Autoencoder',
                                 encoder=MLPModel(model_id='encoder', mlp_blocks=[hidden_block, output_block]),
                                 latent_dim=4)
            logging.info(vae_model)

            filename = '/users/patricknicolas/dev/geometric_learning/data/heart_diseases.csv'
            df = pd.read_csv(filename)
            features_df = df[features]
            batch_size = 4
            train_eval_split_ratio = 0.85
            dataset_loader = UnlabeledLoader(batch_size, train_eval_split_ratio)
            train_loader, eval_loader = dataset_loader.from_dataframe(features_df)

            hyper_parameters = HyperParams(
                lr=0.001,
                momentum=0.95,
                epochs=8,
                optim_label='adam',
                batch_size=8,
                loss_function=nn.MSELoss(),
                drop_out=0.0,
                train_eval_ratio=0.9)

            metric_labels = {
                Metric.accuracy_label: BuiltInMetric(MetricType.Accuracy, True)
            }
            network = VAETraining(hyper_parameters, metric_labels)
            network.train(vae_model.id, vae_model, train_loader, eval_loader)
        except (AssertionError | VAEException | MLPException| TrainingException) as e:
            logging.error(e)
            self.assertTrue(False)


    @staticmethod
    def load_dataset(root_path: AnyStr,
                     exec_config: ExecConfig,
                     batch_size: int,
                     image_size: int) -> (DataLoader, DataLoader):
        train_dataset, test_dataset = VAETest.__extract_datasets(root_path, image_size)

        # If we are experimenting with a subset of the data set for memory usage
        train_dataset, test_dataset = exec_config.apply_sampling(train_dataset,  test_dataset)

        # Create DataLoaders for batch processing
        train_loader, test_loader = exec_config.apply_optimize_loaders(batch_size, train_dataset, test_dataset)
        return train_loader, test_loader

    @staticmethod
    def __extract_datasets(root_path: AnyStr, size_image: int) -> (Dataset, Dataset):
        """
        Extract the training data and labels and test data and labels for this convolutional network.
        @param root_path: Root path to CIFAR10 data
        @type root_path: AnyStr
        @return Tuple (train data, labels, test data, labels)
        @rtype Tuple[torch.Tensor]
        """
        from torchvision.datasets import MNIST
        import torchvision.transforms as transforms
        from dl.model import GrayscaleToRGB

        transform = transforms.Compose([
            transforms.Resize(size =(size_image, size_image), interpolation=transforms.InterpolationMode.BILINEAR),
            GrayscaleToRGB(),
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            # Normalize with mean and std for RGB channels
            transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(0.5, 0.5, 0.5))
        ]) if size_image > 0 else transforms.Compose([
            GrayscaleToRGB(),
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            # Normalize with mean and std for RGB channels
            transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(0.5, 0.5, 0.5))
        ])

        train_dataset = MNIST(
            root=root_path,  # Directory to store the dataset
            train=True,  # Load training data
            download=True,  # Download if not already present
            transform=transform  # Apply transformations
        )

        test_dataset = MNIST(
            root=root_path,  # Directory to store the dataset
            train=False,  # Load test data
            download=True,  # Download if not already present
            transform=transform  # Apply transformations
        )
        return train_dataset, test_dataset


if __name__ == '__main__':
    unittest.main()
