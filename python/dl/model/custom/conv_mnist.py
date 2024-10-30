__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import List, AnyStr, Tuple, NoReturn
import torch.nn as nn
import torch
from dl.model.custom.base_mnist import BaseMnist
from dl.model.convmodel import ConvModel
from dl.block.convblock import ConvBlock
from dl.block.ffnnblock import FFNNBlock
from dl.block.builder.conv2dblockbuilder import Conv2DBlockBuilder
import logging
logger = logging.getLogger('dl.model.custom.ConvMNIST')

__all__ = ['BaseMnist', 'ConvMNIST']


class ConvMNIST(BaseMnist):
    id = 'Convolutional_MNIST'

    def __init__(self,
                 input_size: int,
                 in_channels: List[int],
                 kernel_size: List[int],
                 padding_size: List[int],
                 stride_size: List[int],
                 max_pooling_kernel: int,
                 out_channels: int,
                 activation: nn.Module) -> None:

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
                activation=activation,
                bias=has_bias)

            input_dim = conv_2d_block_builder.get_conv_layer_out_shape()
            conv_blocks.append(ConvBlock(str(idx+1), conv_2d_block_builder))

        conv_output_shape = conv_blocks[len(conv_blocks)-1].compute_out_shapes()
        ffnn_input_shape = out_channels * conv_output_shape[0] * conv_output_shape[1]
        ffnn_block1 = FFNNBlock.build(block_id='hidden',
                                      in_features=ffnn_input_shape,
                                      out_features = 128,
                                      activation=nn.ReLU())
        ffnn_block2 = FFNNBlock.build(block_id='output',
                                      in_features=128,
                                      out_features = BaseMnist.num_classes,
                                      activation=nn.Softmax(dim=1))
        conv_model = ConvModel(ConvMNIST.id, conv_blocks, ffnn_blocks=[ffnn_block1, ffnn_block2])
        super(ConvMNIST, self).__init__(conv_model)

    def show_conv_weights_shape(self) -> NoReturn:
        import torch

        for idx, conv_block in enumerate(self.model.conv_blocks):
            conv_modules_weights: Tuple[torch.Tensor] = conv_block.get_modules_weights()
            print(f'\nConv. layer #{idx} shape: {conv_modules_weights[0].shape}')

    def _extract_datasets(self, root_path: AnyStr) ->(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
             Extract the training data and labels and test data and labels for this convolutional network.
             @param root_path: Root path to MNIST dataset
             @type root_path: AnyStr
             @return Tuple (train data, labels, test data, labels)
             @rtype Tuple[torch.Tensor]
        """
        from dl.training.neuralnet import NeuralNet

        _, torch_device = NeuralNet.get_device()

        train_data = torch.load(f'{root_path}/{BaseMnist.default_training_file}')
        train_features = train_data[0].unsqueeze(dim=1).float().to(torch_device)
        train_labels = torch.nn.functional.one_hot(train_data[1], num_classes=10).float().to(torch_device)

        test_data = torch.load(f'{root_path}/{BaseMnist.default_test_file}')
        test_features = test_data[0].unsqueeze(dim=1).float().to(torch_device)
        test_labels = torch.nn.functional.one_hot(test_data[1], num_classes=10).float().to(torch_device)

        return train_features, train_labels, test_features, test_labels
