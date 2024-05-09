import unittest
import torch.nn as nn
from dl.block.convblock import ConvBlock
from dl.block.ffnnblock import FFNNBlock
from dl.model.convmodel import ConvModel


class ConvModelTest(unittest.TestCase):
    def test_init(self):
        model_id = 'conv_model_1'
        in_channels = 64
        in_channels_2 = 32
        out_channels = 16
        conv_block_1 = ConvModelTest.__create_conv_block(2, in_channels, in_channels_2)
        conv_block_2 = ConvModelTest.__create_conv_block(2, in_channels_2, out_channels)
        ffnn_block_1 = FFNNBlock.build('hidden', out_channels, 4, nn.ReLU())
        ffnn_block_2 = FFNNBlock.build('output', 4, 4, nn.ReLU())
        conv_model = ConvModel(model_id, [conv_block_1, conv_block_2], [ffnn_block_1, ffnn_block_2])
        print(repr(conv_model))

    def test_

    @staticmethod
    def __create_conv_block(dimension: int, in_channels: int, out_channels: int) -> ConvBlock:
        is_batch_normalization = True
        max_pooling_kernel = 2
        activation = nn.Tanh()
        kernel_size = 3
        has_bias = False
        stride = 1
        padding = 1
        return ConvBlock(
            dimension,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            is_batch_normalization,
            max_pooling_kernel,
            activation,
            bias=has_bias)


if __name__ == '__main__':
    unittest.main()