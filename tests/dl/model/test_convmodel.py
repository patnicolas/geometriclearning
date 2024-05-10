import unittest
import torch.nn as nn
from dl.block.convblock import ConvBlock
from dl.block.ffnnblock import FFNNBlock
from dl.model.convmodel import ConvModel
from typing import Tuple


class ConvModelTest(unittest.TestCase):
    def test_init(self):
        model_id = 'conv_model_2d'
        in_channels = 68
        kernel_size = (4, 4)
        in_channels_2 = 16
        kernel_size_2 = (2, 2)
        out_channels = 6
        conv_block_1 = ConvModelTest.__create_conv_block_2(in_channels, in_channels_2, kernel_size)
        conv_block_2 = ConvModelTest.__create_conv_block_2(in_channels_2, out_channels, kernel_size_2)
        ffnn_block_1 = FFNNBlock.build('hidden', out_channels, 4, nn.ReLU())
        ffnn_block_2 = FFNNBlock.build('output', 4, 4, nn.ReLU())
        conv_model = ConvModel(model_id, [conv_block_1, conv_block_2], [ffnn_block_1, ffnn_block_2])
        print(repr(conv_model))


    @staticmethod
    def __create_conv_block_2(in_channels: int, out_channels: int, kernel_size: Tuple[int, int]) -> ConvBlock:
        is_batch_normalization = True
        max_pooling_kernel = 2
        activation = nn.Tanh()
        has_bias = False
        stride = (2, 2)
        padding = (2, 2)
        return ConvBlock(
            2,
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

    """       
    out_channels = 15
            kernel_size = (4, 4)
    input_channels = 64
            is_batch_normalization = True
            max_pooling_kernel = 2
            activation = nn.Tanh()
            has_bias = False
            stride = (2, 2)
            padding = (2, 2)
    """