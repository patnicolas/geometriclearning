import unittest
import torch.nn as nn
from dl.block.deconvblock import DeConvBlock
from dl.model.deconvmodel import DeConvModel
from dl.dlexception import DLException
from typing import Tuple


class DeConvModelTest(unittest.TestCase):

    def test_init(self):
        model_id = 'de_conv_model_2d'
        in_channels = 17
        kernel_size = (4, 4)
        in_channels_2 = 16
        kernel_size_2 = (2, 2)
        out_channels = 60
        try:
            de_conv_block_1 = DeConvModelTest.__create_de_conv_block_2(in_channels, in_channels_2, kernel_size)
            de_conv_block_2 = DeConvModelTest.__create_de_conv_block_2(in_channels_2, out_channels, kernel_size_2)
            de_conv_model = DeConvModel.build(model_id, [de_conv_block_1, de_conv_block_2])
            self.assertTrue(de_conv_model.out_channels == out_channels)
            print(repr(de_conv_model))
            assert True
        except DLException as e:
            assert False

    def test_init_failed(self):
        model_id = 'de_conv_model_2d'
        in_channels = 17
        kernel_size = (4, 4)
        in_channels_2 = 16
        kernel_size_2 = (2, 2)
        out_channels = 64
        try:
            de_conv_block_1 = DeConvModelTest.__create_de_conv_block_2(in_channels, in_channels_2, kernel_size)
            de_conv_block_2 = DeConvModelTest.__create_de_conv_block_2(in_channels_2, out_channels, kernel_size_2)
            de_conv_model = DeConvModel.build(model_id, [de_conv_block_1, de_conv_block_2])
            self.assertTrue(de_conv_model.out_channels == out_channels)
            print(repr(de_conv_model))
            assert False
        except DLException as e:
            assert True

    @staticmethod
    def __create_de_conv_block_2(in_channels: int, out_channels: int, kernel_size: Tuple[int, int]) -> DeConvBlock:
        is_batch_normalization = True
        activation = nn.Tanh()
        has_bias = False
        stride = (2, 2)
        padding = (2, 2)
        return DeConvBlock(
            2,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            is_batch_normalization,
            activation,
            bias=has_bias)
