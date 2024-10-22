__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List, AnyStr, Tuple, NoReturn
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
        input_dim = (input_size, input_size)
        for idx in range(len(in_channels)):
            is_batch_normalization = True
            has_bias = False
            conv_2d_block_builder = Conv2DBlockBuilder(
                in_channels=in_channels[idx],
                out_channels=in_channels[idx+1] if idx < len(in_channels)-1 else out_channels,
                input_size=input_dim,
                kernel_size=(kernel_size[idx], kernel_size[idx]),
                stride=(stride_size[idx], stride_size[idx]),
                padding=(padding_size[idx], padding_size[idx]),
                batch_norm=is_batch_normalization,
                max_pooling_kernel=max_pooling_kernel,
                activation=nn.ReLU(),
                bias=has_bias)
            # conv_2d_block_builder.get_pool_out_shape()
            input_dim = conv_2d_block_builder.get_conv_layer_out_shape()
            conv_blocks.append(ConvBlock(str(idx+1), conv_2d_block_builder))

        num_classes = 10
        conv_output_shape = conv_blocks[len(conv_blocks)-1].compute_out_shapes()
        ffnn_input_shape = out_channels * conv_output_shape[0] * conv_output_shape[1]
        ffnn_block = FFNNBlock.build('hidden', ffnn_input_shape, num_classes, nn.ReLU())
        self.conv_model = ConvModel(ConvMNIST.id, conv_blocks, [ffnn_block])

    def show_conv_weights_shape(self) -> NoReturn:
        import torch

        for idx, conv_block in enumerate(self.conv_model.conv_blocks):
            conv_modules_weights: Tuple[torch.Tensor] = conv_block.get_modules_weights()
            print(f'\nConv. layer #{idx} shape: {conv_modules_weights[0].shape}')

    def __repr__(self) -> AnyStr:
        return repr(self.conv_model)

    def train_(self, root_path: AnyStr, is_testing: bool):
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

        train_data_loader, test_data_loader = ConvMNIST.__load_dataset(root_path, is_testing)
        network(train_data_loader, test_data_loader)


    @staticmethod
    def __load_dataset(root_path: AnyStr, is_testing: bool) -> (DataLoader, DataLoader):
        import torch

        if is_testing:
            train_features = torch.randn(640, 1, 28, 28)
            train_labels = torch.randn(640)
            test_features = torch.randn(64, 1, 28, 28)
            test_labels = torch.randn(64)
        else:
            train_data = torch.load(f'{root_path}/processed/training.pt')
            train_features = train_data[0]
            train_labels = train_data[1]
            test_data = torch.load(f'{root_path}/processed/test.pt')
            test_features = test_data[0]
            test_labels = test_data[1]

        train_dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)

        # Create DataLoaders for batch processing
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

        # Check the first batch of data
        for images, labels in train_loader:
            print(f'Image batch shape: {images.shape}')
            print(f'Label batch shape: {labels.shape}')
            break

        return train_loader, test_loader
