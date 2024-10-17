__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List, AnyStr, Tuple
import torch.nn as nn
from dl.model.convmodel import ConvModel
from torch.utils.data import DataLoader, TensorDataset
from dl.block.convblock import ConvBlock
from dl.block.ffnnblock import FFNNBlock
from dl.training.neuralnet import NeuralNet
from dl.training.earlystoplogger import EarlyStopLogger
from dl.block.builder.conv2dblockbuilder import Conv2DBlockBuilder
from metric.metric import Metric
from plots.plotter import PlotterParameters
from dl.training.hyperparams import HyperParams
from metric.builtinmetric import BuiltInMetric, MetricType
import logging
logger = logging.getLogger('dl.model.custom.ConvMNIST')


class ConvMNIST(object):
    id = 'Convolutional MNIST'

    def __init__(self,
                 input_size: int,
                 in_channels: List[int],
                 kernel_size: List[int],
                 padding_size: List[int],
                 stride_size: List[int],
                 max_pooling_kernel: int,
                 out_channels: int) -> None:

        import torch
        conv_blocks = []
        for idx in range(len(in_channels)):
            is_batch_normalization = True
            has_bias = False
            conv_2d_block_builder = Conv2DBlockBuilder(
                in_channels[idx],
                in_channels[idx+1] if idx < len(in_channels)-1 else out_channels,
                (input_size, input_size),
                (kernel_size[idx], kernel_size[idx]),
                (stride_size[idx], stride_size[idx]),
                (padding_size[idx], padding_size[idx]),
                is_batch_normalization,
                max_pooling_kernel,
                nn.ReLU(),
                has_bias)
            conv_blocks.append(ConvBlock(conv_2d_block_builder))

        for idx, conv_block in enumerate(conv_blocks):
            conv_modules_weights: Tuple[torch.Tensor] = conv_block.get_modules_weights()
            print(f'\nConv. layer #{idx} --------- \n{conv_modules_weights[0].shape}')

        num_classes = 10
        conv_output_shape = conv_blocks[len(conv_blocks)-1].compute_out_shapes()
        ffnn_input_shape = out_channels * conv_output_shape[0] * conv_output_shape[1]
        ffnn_block = FFNNBlock.build('hidden', ffnn_input_shape, num_classes, nn.ReLU())
        self.conv_model = ConvModel(ConvMNIST.id, conv_blocks, [ffnn_block])

    def forward(self, x):
        import torch
        import torch.nn.functional as F
        blk = self.conv_model.conv_blocks

        x = blk[0](x)
        x = F.relu(x)
        x = blk[1](x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        blk = self.conv_model.ffnn_blocks
        x = blk[0](x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output

    def __repr__(self) -> AnyStr:
        return repr(self.conv_model)

    def train_(self, root_path: AnyStr):
        patience = 2
        min_diff_loss = -0.001
        early_stopping_enabled = True
        early_stop_logger = EarlyStopLogger(patience, min_diff_loss, early_stopping_enabled)
        metric_labels = {
            Metric.accuracy_label: BuiltInMetric(MetricType.Accuracy, True),
            Metric.precision_label: BuiltInMetric(MetricType.Precision, True)
        }
        parameters = [PlotterParameters(0, x_label='x', y_label='y', title=label, fig_size=(11, 7))
                      for label, _ in metric_labels.items()]
        hyper_parameters = HyperParams(
            lr=0.001,
            momentum=0.95,
            epochs=8,
            optim_label='adam',
            batch_size=64,
            loss_function=nn.BCELoss(),
            drop_out=0.2,
            train_eval_ratio=0.9,
            normal_weight_initialization = True)
        network = NeuralNet(
            self.conv_model,
            hyper_parameters,
            early_stop_logger,
            metric_labels,
            parameters)
        train_data_loader, test_data_loader = ConvMNIST.__load_dataset(root_path)

        network(train_data_loader, test_data_loader)



    @staticmethod
    def __load_dataset(root_path: AnyStr) -> (DataLoader, DataLoader):
        import torch

        """
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std dev
        ])
        """

        train_data = torch.load(f'{root_path}/processed/training.pt')
        train_features = train_data[0]
        train_labels = train_data[1]
        train_dataset = TensorDataset(train_features, train_labels)

        test_data = torch.load(f'{root_path}/processed/test.pt')
        test_features = test_data[0]
        test_labels = test_data[1]
        test_dataset = TensorDataset(test_features, test_labels)

        # Create DataLoaders for batch processin
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

        # Check the first batch of data
        for images, labels in train_loader:
            print(f'Image batch shape: {images.shape}')
            print(f'Label batch shape: {labels.shape}')
            break

        return train_loader, test_loader
