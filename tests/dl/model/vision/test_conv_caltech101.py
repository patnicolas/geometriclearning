import unittest
from dl.model.vision.conv_2D_config import Conv2DConfig, ConvLayer2DConfig
from dl.model.vision.conv_caltech101 import ConvCaltech101
from dl.training.hyper_params import HyperParams
import torch.nn as nn


class ConvCaltech101Test(unittest.TestCase):
    @unittest.skip('Ignore')
    def test_init(self):
        conv_caltech_101 = ConvCaltech101Test.create_conv_net()
        print(repr(conv_caltech_101))

    @unittest.skip('Ignore')
    def test_mul(self):
        import torch
        x: torch.Tensor = torch.rand(128*128*3)
        x = x.reshape([3, 128, 128])
        mean = torch.Tensor([[0.5], [0.5], [0.5]])
        std = torch.Tensor([[0.5], [0.5], [0.5]])
        sh = x.shape
        t = x.sub_(mean)
        t = t.div_(std)
        print(t.shape)

    def test_train(self):
        conv_caltech_101 = ConvCaltech101Test.create_conv_net()
        root_path = '../../../../data/caltech-101'
        lr = 0.001

        hyper_parameters = HyperParams(
            lr=lr,
            momentum=0.89,
            epochs=2,
            optim_label='adam',
            batch_size=4,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.25,
            train_eval_ratio=0.9,
            normal_weight_initialization=False)

        conv_caltech_101.do_train(
            root_path,
            hyper_parameters,
            metric_labels=['Precision', 'Recall'],
            plot_title=f'Caltech_101_{lr}'
        )

    @staticmethod
    def create_conv_net() -> ConvCaltech101:
        _id = 'Caltech101'
        input_size = 128
        max_pooling_kernel = 2
        out_channels = 512
        activation = nn.ReLU()
        ffnn_out_features = [512, 256]
        num_classes = 101
        batch_size = 4
        subset_size = 1000

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
        return ConvCaltech101(
            conv_2D_config=conv_2D_config,
            data_batch_size=batch_size,
            resize_image=input_size,
            subset_size=subset_size)
