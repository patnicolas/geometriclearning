import unittest
import logging

from dl import ConvException
from dl.block.conv.conv_2d_block import Conv2dBlock
import torch.nn as nn
import os
import python
from python import SKIP_REASON


class Conv2dBlockTest(unittest.TestCase):

    def test_init_1(self):
        try:
            conv_layer_module = nn.Conv2d(in_channels=1,
                                          out_channels=32,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=(1, 1),
                                          bias=False)

            conv_2d_block = Conv2dBlock(block_id='My_conv_2d',
                                        conv_layer_module=conv_layer_module,
                                        batch_norm_module=nn.BatchNorm2d(32),
                                        activation_module=nn.ReLU(),
                                        max_pooling_module=nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                                        drop_out_module=nn.Dropout2d(0.2))
            logging.info(str(conv_2d_block))
            attributes = {}
            conv_2d_block.validate(attributes)
            self.assertTrue(len(conv_2d_block.attributes) == 5)
        except (AssertionError | ConvException) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_init_2(self):
        try:
            conv_layer_module = nn.Conv2d(in_channels=1,
                                          out_channels=32,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=(1, 1),
                                          bias=False)
            conv_2d_block = Conv2dBlock(block_id='My_conv_2d',
                                        conv_layer_module=conv_layer_module,
                                        batch_norm_module=nn.BatchNorm2d(32),
                                        activation_module=nn.ReLU(),
                                        max_pooling_module=nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                                        drop_out_module=nn.Dropout2d(0.2))
            conv_2d_block.validate()
            logging.info(f'{conv_2d_block=}')
            self.assertTrue(conv_2d_block.attributes is None)
        except (AssertionError | ConvException)  as e:
            logging.error(e)
            self.assertTrue(False)


    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_3(self):
        try:
            conv_2d_block = Conv2dBlock.build_from_params(block_id='My_conv_2d',
                                                          in_channels=1,
                                                          out_channels=32,
                                                          kernel_size=(3, 3),
                                                          stride=(1, 1),
                                                          padding=(0, 0),
                                                          batch_norm=True,
                                                          max_pooling_kernel=2,
                                                          activation=nn.ReLU(),
                                                          bias=False,
                                                          drop_out=0.0)
            logging.info(f'{conv_2d_block=}')
            self.assertTrue(len(conv_2d_block.attributes) == 0)
        except (AssertionError | ConvException)  as e:
            logging.error(e)
            self.assertTrue(False)


    def test_init_4(self):
        try:
            block_attributes = {
                'block_id': 'my_block',
                'in_channels': 64,
                'out_channels': 128,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (2, 2),
                'bias': True,
                'batch_norm': nn.BatchNorm2d(32),
                'activation': nn.ReLU(),
                'max_pooling': nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                'dropout_ratio': 0.3
            }
            conv_2d_block = Conv2dBlock.build(block_attributes)
            logging.info(f'{conv_2d_block=}')
            self.assertTrue(len(conv_2d_block.modules_list) == 5)
        except (AssertionError | ConvException)  as e:
            logging.error(e)
            self.assertTrue(False)


    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_transpose(self):
        try:
            conv_2d_block = Conv2dBlock.build_from_params(block_id='My_conv_2d',
                                                          in_channels=1,
                                                          out_channels=32,
                                                          kernel_size=(3, 3),
                                                          stride=(1, 1),
                                                          padding=(0, 0),
                                                          batch_norm=True,
                                                          max_pooling_kernel=2,
                                                          activation=nn.ReLU(),
                                                          bias=False,
                                                          drop_out=0.2)
            logging.info(conv_2d_block.get_attributes())
            de_conv_2d_block = conv_2d_block.transpose(output_activation=nn.Sigmoid())
            logging.info(f'{de_conv_2d_block=}')
            self.assertTrue(str(conv_2d_block.attributes['activation']) == 'Sigmoid()')
        except (AssertionError | ConvException)  as e:
            logging.error(e)
            self.assertTrue(False)
