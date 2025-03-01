import unittest

from dl.block.conv.conv_2d_block import Conv2dBlock
import torch.nn as nn


class Conv2dBlockTest(unittest.TestCase):
    def test_init_1(self):
        conv_layer_module = nn.Conv2d(in_channels=1,
                                      out_channels=32,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=(1, 1),
                                      bias=False)

        conv_2d_block = Conv2dBlock('My_conv_2d',
                                    conv_layer_module=conv_layer_module,
                                    max_pooling_module= nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                                    deconvolution_enabled=False,
                                    batch_norm_module=nn.BatchNorm2d(32),
                                    activation_module=nn.ReLU(),
                                    drop_out_module=nn.Dropout2d(0.2))
        print(str(conv_2d_block))
        self.assertTrue(len(conv_2d_block.attributes) == 0)

    @unittest.skip('Ignore')
    def test_init_2(self):
        conv_2d_block = Conv2dBlock.build(block_id='My_conv_2d',
                                          in_channels=1,
                                          out_channels=32,
                                          kernel_size=(3, 3),
                                          deconvolution_enabled=False,
                                          stride=(1, 1),
                                          padding=(0, 0),
                                          batch_norm=True,
                                          max_pooling_kernel=2,
                                          activation=nn.ReLU(),
                                          bias=False,
                                          drop_out=0.0)
        print(str(conv_2d_block))
        self.assertTrue(len(conv_2d_block.attributes) == 0)

    @unittest.skip('Ignore')
    def test_transpose(self):
        conv_2d_block = Conv2dBlock.build(block_id='My_conv_2d',
                                          in_channels=1,
                                          out_channels=32,
                                          kernel_size=(3, 3),
                                          deconvolution_enabled=True,
                                          stride=(1, 1),
                                          padding=(0, 0),
                                          batch_norm=True,
                                          max_pooling_kernel=2,
                                          activation=nn.ReLU(),
                                          bias=False,
                                          drop_out=0.2)
        print(conv_2d_block.get_attributes())
        de_conv_2d_block = conv_2d_block.transpose(output_activation=nn.Sigmoid())
        print(str(de_conv_2d_block))
        self.assertTrue(str(conv_2d_block.attributes['activation']) == 'Sigmoid()')