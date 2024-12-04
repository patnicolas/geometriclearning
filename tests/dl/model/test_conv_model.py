import unittest
import torch.nn as nn
from dl.block.conv_block import ConvBlock
from dl.block.conv_2d_block import Conv2DBlock
from dl.block.ffnn_block import FFNNBlock
from dl.model.conv_model import ConvModel
from dl import ConvException
import logging


class ConvModelTest(unittest.TestCase):

    def test_mnist_small(self):
        try:
            conv_2d_block_1 = Conv2DBlock.build(block_id='conv_1',
                                                in_channels=1,
                                                out_channels=8,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1,1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            conv_2d_block_2 = Conv2DBlock.build(block_id='conv_2',
                                                in_channels=8,
                                                out_channels=16,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1,1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            num_classes = 10
            ffnn_block_1 = FFNNBlock.build(block_id='hidden',
                                           in_features=0,
                                           out_features=num_classes,
                                           activation=nn.Softmax(dim=1))
            conv_model = ConvModel(model_id='MNIST',
                                   input_size=(28, 28),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2],
                                   ffnn_blocks=[ffnn_block_1])
            print(repr(conv_model))
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    def test_mnist_large(self):
        try:
            conv_2d_block_1 = Conv2DBlock.build(block_id='conv_1',
                                                in_channels=1,
                                                out_channels=8,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            conv_2d_block_2 = Conv2DBlock.build(block_id='conv_2',
                                                in_channels=8,
                                                out_channels=16,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            conv_2d_block_3 = Conv2DBlock.build(block_id='conv_3',
                                                in_channels=16,
                                                out_channels=32,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            ffnn_block_1 = FFNNBlock.build(block_id='hidden',
                                           in_features=0,
                                           out_features=64,
                                           activation=nn.ReLU())
            num_classes = 10
            ffnn_block_2 = FFNNBlock.build(block_id='output',
                                           in_features=64,
                                           out_features=num_classes,
                                           activation=nn.Softmax(dim=1))
            conv_model = ConvModel(model_id='MNIST',
                                   input_size=(28, 28),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2, conv_2d_block_3],
                                   ffnn_blocks=[ffnn_block_1, ffnn_block_2])
            print(repr(conv_model))
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    def test_invert(self):
        try:
            conv_2d_block_1 = Conv2DBlock.build(block_id='conv_1',
                                                in_channels=1,
                                                out_channels=8,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            conv_2d_block_2 = Conv2DBlock.build(block_id='conv_2',
                                                in_channels=8,
                                                out_channels=16,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            conv_2d_block_3 = Conv2DBlock.build(block_id='conv_3',
                                                in_channels=16,
                                                out_channels=32,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            conv_model = ConvModel(model_id='MNIST',
                                   input_size=(28, 28),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2, conv_2d_block_3],
                                   ffnn_blocks=None)
            print(f'\nConv modules:---\n{repr(conv_model)}')
            de_conv_model = conv_model.transpose(extra=nn.Sigmoid())
            print(f'\nDe conv modules: ----\n{repr(de_conv_model)}')
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
