import unittest
import torch.nn as nn
from dl.block.convblock import ConvBlock
from dl.block.ffnnblock import FFNNBlock
from dl.model.convmodel import ConvModel
from dl.block import ConvException
from dl.block.builder.conv2dblockbuilder import Conv2DBlockBuilder
import logging


class ConvModelTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init(self):
        model_id = 'conv_model_2d'
        in_channels = 68
        kernel_size = (4, 4)
        in_channels_2 = 16
        kernel_size_2 = (2, 2)
        out_channels = 6
        try:
            conv_block_1 = ConvModelTest.__create_conv_block_2(in_channels, in_channels_2, kernel_size)
            conv_block_2 = ConvModelTest.__create_conv_block_2(in_channels_2, out_channels, kernel_size_2)
            ffnn_block_1 = FFNNBlock.build('hidden', out_channels, 4, nn.ReLU())
            ffnn_block_2 = FFNNBlock.build('output', 4, 4, nn.ReLU())
            conv_model = ConvModel(model_id, [conv_block_1, conv_block_2], [ffnn_block_1, ffnn_block_2])
            self.assertTrue(conv_model.has_fully_connected())
            self.assertTrue(conv_model.in_features == in_channels)
            self.assertTrue(conv_model.out_features == out_channels)
            logging.info(repr(conv_model))
            self.assertTrue(True)
        except ConvException as e:
            logging.error(str(e))
            self.assertTrue(True)

    @unittest.skip('Ignore')
    def test_init_2(self):
        model_id = 'conv_model_2d'
        in_channels = 68
        kernel_size = (4, 4)
        in_channels_2 = 16
        kernel_size_2 = (6, 6)
        out_channels = 6
        try:
            conv_block_1 = ConvModelTest.__create_conv_block_2(in_channels, in_channels_2, kernel_size)
            conv_block_2 = ConvModelTest.__create_conv_block_2(in_channels_2, out_channels, kernel_size_2)
            ffnn_block_1 = FFNNBlock.build('hidden', out_channels, 4, nn.ReLU())
            ffnn_block_2 = FFNNBlock.build('output', 4, 4, nn.ReLU())
            conv_model = ConvModel(model_id, [conv_block_1, conv_block_2], [ffnn_block_1, ffnn_block_2])
            logging.info(repr(conv_model))
            self.assertTrue(False)
        except ConvException as e:
            logging.error(str(e))
            self.assertTrue(True)

    def test_mnist(self):
        model_id = 'conv_MNIST_model'
        input_size = 28
        in_channels = 1
        kernel_size = 3
        padding_size = 1
        stride_size = 1
        in_channels_2 = 8
        kernel_size_2 = 3
        out_channels = 16
        num_classes = 10
        try:
            conv_block_1 = ConvModelTest.__create_conv_block_2(
                in_channels,
                in_channels_2,
                input_size,
                kernel_size,
                stride_size,
                padding_size,
                nn.ReLU()
            )
            conv_block_2 = ConvModelTest.__create_conv_block_2(
                in_channels_2,
                out_channels,
                input_size,
                kernel_size_2,
                stride_size,
                padding_size,
                nn.ReLU()
            )
            conv_output_shape = conv_block_2.compute_out_shapes()
            ffnn_input_shape = out_channels*conv_output_shape[0]*conv_output_shape[1]
            ffnn_block_1 = FFNNBlock.build('hidden', ffnn_input_shape, num_classes, nn.ReLU())
            conv_model = ConvModel(model_id, [conv_block_1, conv_block_2], [ffnn_block_1])
            print(repr(conv_model))
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @staticmethod
    def __create_conv_block_2(
            in_channels: int,
            out_channels: int,
            input_size: int,
            kernel_size: int,
            stride_size: int,
            padding_size: int,
            activation: nn.Module) -> ConvBlock:
        is_batch_normalization = True
        max_pooling_kernel = 2
        activation = activation
        has_bias = False
        conv_2d_block_builder = Conv2DBlockBuilder(
            in_channels,
            out_channels,
            (input_size, input_size),
            (kernel_size, kernel_size),
            (stride_size, stride_size),
            (padding_size, padding_size),
            is_batch_normalization,
            max_pooling_kernel,
            activation,
            has_bias)
        return ConvBlock(conv_2d_block_builder)


if __name__ == '__main__':
    unittest.main()
