import unittest
from dl.model.custom.conv_2D_config import Conv2DConfig, ConvLayer2DConfig
from dl.model.custom.conv_cifar10 import ConvCifar10
import torch.nn as nn


class ConvCifar10Test(unittest.TestCase):

    def test_init(self):
        _id = 'test'
        input_size = 28
        max_pooling_kernel = 2
        out_channels = 128
        activation = nn.ReLU()
        ffnn_out_features = [256, 128, 128]
        num_classes = 10

        conv_layer1_config = ConvLayer2DConfig(1, 3, 0, 1)
        conv_layer2_config = ConvLayer2DConfig(32, 3, 0, 1)
        conv_layer3_config = ConvLayer2DConfig(64, 3, 0, 1)
        conv_layer_2D_config = [conv_layer1_config, conv_layer2_config, conv_layer3_config]
        conv_2D_config = Conv2DConfig(_id,
                                      input_size,
                                      conv_layer_2D_config,
                                      max_pooling_kernel,
                                      out_channels,
                                      activation,
                                      ffnn_out_features,
                                      num_classes)
        conv_cifar_10 = ConvCifar10(conv_2D_config)
        print(repr(conv_cifar_10))