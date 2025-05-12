import unittest
import logging
from torch import nn

from dl.block.conv.deconv_2d_block import DeConv2dBlock
from dl import ConvException


class DeConv2dBlockTest(unittest.TestCase):

    def test_init_de_conv1(self):
        try:
            de_conv_2d_block = DeConv2dBlock(block_id='my_deconvolution_model',
                                             de_conv_2d_module=nn.ConvTranspose2d(in_channels=32,
                                                                                  out_channels=64,
                                                                                  kernel_size=(3, 3),
                                                                                  stride=(1, 1),
                                                                                  padding=(1, 1)),
                                             batch_norm_module=nn.BatchNorm2d(64),
                                             activation_module=nn.ReLU(),
                                             drop_out_module=nn.Dropout2d(0.3))
            logging.info(repr(de_conv_2d_block))
            self.assertTrue(de_conv_2d_block.de_conv_2d_module.out_channels == 64)
        except ConvException as e:
            logging.info(str(e))
            self.assertTrue(False)

    def test_init_de_conv2(self):
        try:
            block_attributes = {
                'block_id': 'my_model',
                'in_channels': 32,
                'out_channels': 64,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (1, 1),
                'batch_norm': nn.BatchNorm2d(64),
                'activation': nn.ReLU(),
                'dropout_ratio': 0.3
            }
            de_conv_2d_block = DeConv2dBlock.build(block_attributes)
            logging.info(repr(de_conv_2d_block))
            self.assertTrue(str(de_conv_2d_block.conv_block_config.activation_module) == 'Sigmoid()')
        except ConvException as e:
            logging.info(str(e))
            self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()