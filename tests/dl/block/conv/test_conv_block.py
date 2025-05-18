import unittest
from torch import nn
import logging
from dl import ConvException
from dl.block.conv.conv_2d_block import Conv2dBlock
from dl.block.conv.conv_3d_block import Conv3dBlock


class ConvBlockTest(unittest.TestCase):

    def test_init_conv1(self):
        try:
            conv_block = ConvBlockTest.__create_conv_block(dimension=1, out_channels=33)
            self.assertTrue(conv_block.conv_block_config.out_channels == 33)
            logging.info(f'{conv_block=}')
        except ConvException as e:
            logging.info(str(e))
            self.assertTrue(False)

    def test_init_conv2(self):
        try:
            conv_block = ConvBlockTest.__create_conv_block(dimension=2, out_channels=19)
            self.assertTrue(conv_block.conv_block_config.out_channels == 19)
            logging.info(repr(conv_block))
            self.assertTrue(True)
        except ConvException as e:
            logging.info(str(e))
            self.assertTrue(False)

    @staticmethod
    def __create_conv_block(dimension: int, out_channels: int) -> Conv2dBlock | Conv3dBlock:
        if dimension == 2:
            return Conv2dBlock.build_from_params(in_channels=68,
                                                 out_channels=out_channels,
                                                 kernel_size=(2, 2),
                                                 stride=(2, 2),
                                                 padding=(2, 2),
                                                 batch_norm=True,
                                                 max_pooling_kernel=2,
                                                 activation=nn.Tanh(),
                                                 bias=False,
                                                 drop_out=0.0)
        elif dimension == 3:
            return Conv3dBlock.build_from_params(in_channels=65,
                                                 out_channels=out_channels,
                                                 kernel_size=(3, 3, 3),
                                                 stride=(3, 3, 3),
                                                 padding=(3, 3, 3),
                                                 batch_norm=True,
                                                 max_pooling_kernel=2,
                                                 activation=nn.Tanh(),
                                                 bias=False,
                                                 drop_out=0.0)
        else:
            raise ConvException(f'{dimension=} is not supported')


if __name__ == '__main__':
    unittest.main()