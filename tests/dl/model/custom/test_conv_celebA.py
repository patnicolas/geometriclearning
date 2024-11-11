import unittest
from dl.model.custom.conv_2D_config import Conv2DConfig, ConvLayer2DConfig
from dl.model.custom.conv_celebA import ConvCelebA
from dl.training.hyper_params import HyperParams
import torch.nn as nn


class ConvCelebATest(unittest.TestCase):

    def test_train(self):
        conv_celebA = ConvCelebATest.create_conv_net()
        print(repr(conv_celebA))
        root_path = '../../../../data/'
        # root_path = './data'
        lr = 0.001

        hyper_parameters = HyperParams(
            lr=lr,
            momentum=0.89,
            epochs=10,
            optim_label='adam',
            batch_size=16,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.15,
            train_eval_ratio=0.9,
            normal_weight_initialization=False)

        conv_celebA.do_train(
            root_path,
            hyper_parameters,
            metric_labels=['Precision', 'Recall'],
            plot_title=f'CIFAR10_{lr}'
        )

    @staticmethod
    def create_conv_net() -> ConvCelebA:
        _id = 'CelebA'
        input_size = 32
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
        return ConvCelebA(conv_2D_config)