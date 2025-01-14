import unittest
from dl.block.ffnn_block import FFNNBlock
from dl.model.ffnn_model import FFNNModel
from dl.model.vae_model import VAEModel
from dl.model.conv_model import ConvModel
from dl.block.conv.conv_2d_block import Conv2DBlock
from dl.block.conv.conv_block_config import ConvBlockConfig
from dataset.unlabeled_loader import UnlabeledLoader
from dl.training.exec_config import ExecConfig
from dl.training.vae_training import VAETraining
from dl.training.hyper_params import HyperParams
from dl.training.training_summary import TrainingSummary
from metric.metric import Metric, MetricType
from metric.built_in_metric import BuiltInMetric
from dl import ConvException, VAEException
from torch.utils.data import DataLoader, Dataset
from typing import AnyStr
import torch.nn as nn


class VAETest(unittest.TestCase):

    def test_init(self):
        features = ['age', 'sex', 'chest pain type', 'cholesterol', 'fasting blood sugar', 'max heart rate',
                    'exercise angina', 'ST slope']
        hidden_block = FFNNBlock.build(block_id='hidden',
                                       layer=nn.Linear(in_features=len(features), out_features=4),
                                       activation=nn.ReLU())
        output_block = FFNNBlock.build(block_id='latent',
                                       layer=nn.Linear(in_features=4, out_features=4),
                                       activation=nn.ReLU())
        vae_model = VAEModel(model_id='Autoencoder',
                             encoder=FFNNModel(model_id='encoder', neural_blocks=[hidden_block, output_block]),
                             latent_size=6)
        print(vae_model)

    def test_train_1(self):
        from python.metric.metric import Metric
        from python.metric.built_in_metric import BuiltInMetric, MetricType

        features = ['age', 'sex', 'chest pain type', 'cholesterol', 'fasting blood sugar', 'max heart rate',
                    'exercise angina', 'ST slope']
        hidden_block = FFNNBlock.build(block_id='hidden',
                                       layer=nn.Linear(in_features=len(features), out_features=4),
                                       activation=nn.ReLU())
        output_block = FFNNBlock.build(block_id='latent',
                                       layer=nn.Linear(in_features=4, out_features=4),
                                       activation=nn.ReLU())
        vae_model = VAEModel(model_id='Autoencoder',
                             encoder=FFNNModel(model_id='encoder', neural_blocks=[hidden_block, output_block]),
                             latent_size =4)
        print(vae_model)

        filename = '/users/patricknicolas/dev/geometric_learning/data/heart_diseases.csv'
        df = UnlabeledDataset.data_frame(filename)
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

        training_summary = TrainingSummary(patience=2, min_diff_loss=-0.001, early_stopping_enabled=True)
        metric_labels = {
            Metric.accuracy_label: BuiltInMetric(MetricType.Accuracy, True)
        }
        network = VAETraining(
            hyper_parameters,
            training_summary,
            metric_labels,
            ExecConfig.default('mps'))
        network.train(vae_model.id, vae_model, train_loader, eval_loader)

    def test_train_2(self):
        vae_model =  VAETest.create_mnist_vae_model()
        if vae_model is not None:
            hyper_parameters = HyperParams(
                lr=0.001,
                momentum=0.95,
                epochs=8,
                optim_label='adam',
                batch_size=8,
                loss_function=nn.MSELoss(),
                drop_out=0.0,
                train_eval_ratio=0.9)
            training_summary = TrainingSummary(patience=2, min_diff_loss=-0.001, early_stopping_enabled=True)
            metric_labels = {
                Metric.accuracy_label: BuiltInMetric(MetricType.Accuracy, True)
            }
            vae_training = VAETraining(hyper_params=hyper_parameters,
                                       training_summary=training_summary,
                                       metrics=metric_labels,
                                       exec_config=ExecConfig.default('mps'),
                                       plot_parameters=None)
            train_dataset, eval_dataset = VAETest.load_dataset(root_path= '../../../../data/MNIST',
                                                               exec_config=ExecConfig.default('mps'),
                                                               batch_size=8,
                                                               image_size=28)
            vae_training.train(vae_model.id, vae_model, train_dataset, eval_dataset)

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


    @staticmethod
    def create_mnist_vae_model() -> VAEModel | None:
        try:
            conv_2d_block_config = ConvBlockConfig(
                in_channels=1,
                out_channels=32,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                batch_norm=True,
                max_pooling_kernel=-1,
                activation=nn.ReLU(),
                bias=False)
            conv_block_1 = Conv2DBlock( block_id='Conv1',
                                        conv_block_config=conv_2d_block_config)
            conv_2d_block_config = ConvBlockConfig(
                in_channels=32,
                out_channels=64,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                batch_norm=True,
                max_pooling_kernel=-1,
                activation=nn.ReLU(),
                bias=False)
            conv_block_2 = Conv2DBlock(block_id='Conv2', conv_block_config=conv_2d_block_config)
            conv_model = ConvModel(model_id ='conv_MNIST_model', conv_blocks=[conv_block_1, conv_block_2])
            return VAEModel(model_id='VAE - Mnist',
                            encoder=conv_model,
                            latent_size=64,
                            decoder_out_activation=nn.Sigmoid())
        except ConvException as e:
            print(str(e))
            return None
        except VAEException as e:
            print(str(e))
            return None


if __name__ == '__main__':
    unittest.main()
