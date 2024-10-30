import unittest
from python.dl.model.custom.conv_mnist import ConvMNIST
from python.dl.block import ConvException
from python.dl.dlexception import DLException
import torch.nn as nn
from typing import NoReturn, AnyStr


class ConvMNISTTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init(self):
        input_size = 28
        in_channels = [1, 32]
        kernel_size = [3, 3]
        padding_size = [0, 0]
        stride_size = [1, 1]
        max_pooling_kernel = 2
        out_channels = 64

        try :
            conv_MNIST_instance = ConvMNIST(
                input_size,
                in_channels,
                kernel_size,
                padding_size,
                stride_size,
                max_pooling_kernel,
                out_channels)
            print(repr(conv_MNIST_instance))
            print(conv_MNIST_instance.show_conv_weights_shape())
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_conv_weights(self):
        import torch

        # Define the convolutional layer
        conv_layer = nn.Conv2d(
            in_channels=3,  # Number of input channels (e.g., RGB image has 3 channels)
            out_channels=16,  # Number of output channels (number of filters)
            kernel_size=(4, 2),  # Kernel size (4x2)
            stride=(2, 1)  # Stride (2 along height, 1 along width)
        )

        print(f'Shape conv:\n{conv_layer.weight.data.shape}')

        # Example input: batch of 8 RGB images, each of size 32x32
        input_data = torch.randn(8, 3, 32, 32)

        # Forward pass through the convolutional layer
        output = conv_layer(input_data)
        # Print the shape of the output
        print(output.shape)

    @unittest.skip('Ignore')
    def test_flatten(self):
        import torch

        x = torch.randn(64, 64, 5, 5)
        print(f'\nBefore {x.shape}')
        # mod = nn.Flatten(0, 2)
        y = x.view(-1, 10)
        # x = mod(x)
        print(f'\nAfter {y.shape}')

    def test_train(self):
        lr = 0.0002
        activation = nn.LeakyReLU(negative_slope=0.02)
        ConvMNISTTest.create_network(lr, activation)

    @staticmethod
    def create_network(lr: float, activation: nn.Module) -> NoReturn:
        from dl.training.hyperparams import HyperParams

        input_size = 28
        in_channels = [1, 32, 64]
        kernel_size = [3, 3, 3]
        padding_size = [0, 0, 0]
        stride_size = [1, 1, 1]
        max_pooling_kernel = 2
        out_channels = 128
        root_path = '../../../../data/MNIST'
        try:
            hyper_parameters = HyperParams(
                lr=lr,
                momentum=0.86,
                epochs=16,
                optim_label='adam',
                batch_size=16,
                loss_function=nn.CrossEntropyLoss(),
                drop_out=0.25,
                train_eval_ratio=0.9,
                normal_weight_initialization=False)
            conv_MNIST_instance = ConvMNIST(
                    input_size,
                    in_channels,
                    kernel_size,
                    padding_size,
                    stride_size,
                    max_pooling_kernel,
                    out_channels,
                    activation)
            activation_label = str(activation).strip('()')
            conv_MNIST_instance.do_train(root_path, hyper_parameters, activation_label)
        except ConvException as e:
            print(str(e))
        except AssertionError as e:
            print(str(e))
        except DLException as e:
            print(str(e))


if __name__ == '__main__':
    unittest.main()

