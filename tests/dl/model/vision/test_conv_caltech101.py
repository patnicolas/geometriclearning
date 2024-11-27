import unittest
from dl.model.vision.conv_2d_config import Conv2DConfig, ConvLayer2DConfig
from dl.model.vision.conv_caltech101 import ConvCaltech101
from dl.training.hyper_params import HyperParams
from dl.exception.dl_exception import DLException
import torch.nn as nn


class ConvCaltech101Test(unittest.TestCase):

    def test_init(self):
        import torch
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            x = torch.ones(1, device=mps_device)
            print(x)
        else:
            print("MPS device not found.")

        # conv_caltech_101 = ConvCaltech101Test.__create_conv_net()
        # print(repr(conv_caltech_101))

    @unittest.skip('Ignore')
    def test_train(self):
        from dl.training.exec_config import ExecConfig

        root_path = '../../../../data/caltech-101'
        lr = 0.001

        hyper_parameters = HyperParams(
            lr=lr,
            momentum=0.89,
            epochs=2,
            optim_label='adam',
            batch_size=4,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.2,
            train_eval_ratio=0.9,
            encoding_len=101,  # No one-hot encoding
            normal_weight_initialization=False)

        empty_cache: bool = False
        mix_precision: bool = False
        pin_memory: bool = False
        subset_size: int = 200

        exec_config = ExecConfig(
            empty_cache=empty_cache,
            mix_precision=mix_precision,
            pin_mem=pin_memory,
            subset_size=subset_size,
            monitor_memory=True,
            grad_accu_steps=1,
            device_config=None)

        try:
            conv_caltech_101 = ConvCaltech101Test.__create_conv_net()
            conv_caltech_101.do_train(
                root_path,
                hyper_parameters,
                metric_labels=['Precision', 'Recall'],
                exec_config=exec_config,
                plot_title=f'Caltech_101_{lr}'
            )
            self.assertTrue(True)
        except DLException as e:
            self.assertTrue(False, str(e))

    @unittest.skip('Ignore')
    def test_compare(self):
        import matplotlib.pyplot as plt
        import numpy as np

        # Example data
        categories = [
            'Sample 20%[10]',
            'Memory pinning[7]',
            'Mixed precision[2]',
            'Emptying cache[5]',
            'Accumulating grads[8]'
        ]
        var_MNIST = [39, 8, 23, 5, 15]
        var_Caltech101 = [48, 10, 17, 9, 16]

        # Number of categories
        x = np.arange(len(categories))

        # Bar width
        width = 0.3

        # Create bar plots
        plt.bar(x - width, var_MNIST, width, label='MNIST')
        plt.bar(x, var_Caltech101, width, label='Caltech101')

        # Add labels, title, and legend
        plt.xlabel('Recommendations', fontdict={'family': 'serif', 'size': 14, 'weight': 'bold'} )
        plt.ylabel('Memory reduction (%)', fontdict={'family': 'serif', 'size':14, 'weight': 'bold'} )
        plt.title('Comparative impact of memory reduction recommendations', fontdict={'family': 'serif', 'size': 18, 'weight': 'bold'} )
        plt.xticks(x, categories)
        plt.legend(prop={'family': 'serif', 'size': 15, 'style': 'italic'} )

        # Display the plot
        plt.tight_layout()
        plt.show()

    @staticmethod
    def __create_conv_net() -> ConvCaltech101:
        _id = 'Caltech101'
        input_size = 96
        max_pooling_kernel = 2
        out_channels = 512
        activation = nn.ReLU()
        ffnn_out_features = [512, 256]
        num_classes = 101
        batch_size = 4

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
            resize_image=input_size)
