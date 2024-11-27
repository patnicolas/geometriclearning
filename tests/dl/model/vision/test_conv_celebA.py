import unittest
from dl.model.vision.conv_2d_config import Conv2DConfig, ConvLayer2DConfig
from dl.model.vision.conv_celebA import ConvCelebA
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
            epochs=2,
            optim_label='adam',
            batch_size=8,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.25,
            train_eval_ratio=0.9,
            normal_weight_initialization=False)

        conv_celebA.do_train(
            root_path,
            hyper_parameters,
            metric_labels=['Precision', 'Recall'],
            plot_title=f'CelebA_{lr}'
        )

    @staticmethod
    def create_conv_net() -> ConvCelebA:
        _id = 'CelebA'
        input_size = 96
        max_pooling_kernel = 2
        out_channels = 512
        activation = nn.ReLU()
        ffnn_out_features = [512, 256]
        num_classes = 40
        batch_size = 8
        subset_size = 4096

        conv_layer1_config = ConvLayer2DConfig(3, 3, 0, 1)
        conv_layer2_config = ConvLayer2DConfig(64, 3, 0, 1)
        conv_layer3_config = ConvLayer2DConfig(128, 3, 0, 1)
        conv_layer4_config = ConvLayer2DConfig(256, 3, 0, 1)
        conv_layer_2D_config = [
            conv_layer1_config,
            conv_layer2_config,
            conv_layer3_config,
            conv_layer4_config
        ]
        conv_2D_config = Conv2DConfig(_id,
                                      input_size,
                                      conv_layer_2D_config,
                                      max_pooling_kernel,
                                      out_channels,
                                      activation,
                                      ffnn_out_features,
                                      num_classes)
        return ConvCelebA(
            conv_2D_config=conv_2D_config,
            data_batch_size=batch_size,
            resize_image=input_size,
            subset_size=subset_size)
