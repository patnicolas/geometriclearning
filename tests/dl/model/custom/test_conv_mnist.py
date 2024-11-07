import unittest

from dl.training.early_stop_logger import EarlyStopLogger
from python.dl.model.custom.conv_mnist import ConvMNIST
from python.dl.block import ConvException
from python.dl.dl_exception import DLException
import torch.nn as nn
import torch
from typing import NoReturn, AnyStr, List


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

    @unittest.skip('Ignore')
    def test_evaluate_RU_F1(self):
        from plots.plotter import Plotter
        from plots.plotter import PlotterParameters

        elu_f1_scores = ConvMNISTTest.extract_f1_scores('ELU')
        relu_f1_scores = ConvMNISTTest.extract_f1_scores('ReLU')
        leaky_relu_f1_scores = ConvMNISTTest.extract_f1_scores('LeakyReLU')
        gelu_f1_scores = ConvMNISTTest.extract_f1_scores('GELU')
        labels = ['ReLU', 'LeakyReLU', 'ELU', 'GELU']
        Plotter.plot(
            values =[relu_f1_scores, leaky_relu_f1_scores, elu_f1_scores, gelu_f1_scores],
            labels = labels,
            plotter_parameters=PlotterParameters(
                0,
                x_label='Epoch',
                y_label='F1',
                title='Comparison Rectifier Units: F1 score',
                fig_size=(11, 7))
            )

    @unittest.skip('Ignore')
    def test_evaluate_RU_eval_losses(self):
        from plots.plotter import Plotter
        from plots.plotter import PlotterParameters

        labels = ['ReLU', 'LeakyReLU', 'ELU', 'GELU']
        elu_eval_loss = ConvMNISTTest.extract_eval_losses('ELU')
        relu_eval_loss = ConvMNISTTest.extract_eval_losses('ReLU')
        leaky_relu_eval_loss = ConvMNISTTest.extract_eval_losses('LeakyReLU')
        gelu_eval_loss = ConvMNISTTest.extract_eval_losses('GELU')
        Plotter.plot(
            values=[relu_eval_loss, leaky_relu_eval_loss, elu_eval_loss, gelu_eval_loss],
            labels=labels,
            plotter_parameters=PlotterParameters(
                0,
                x_label='Epoch',
                y_label='Evaluation loss',
                title='Comparison Rectifier Units: Evaluation Loss',
                fig_size=(11, 7))
        )


    def test_train(self):
        lr = 0.0006
        # activation = nn.LeakyReLU(negative_slope=0.03)
        # activation = nn.ReLU()
        # activation = nn.ELU()
        activation = nn.GELU()
        ConvMNISTTest.create_and_train_network(lr, activation)

    @staticmethod
    def extract_f1_scores(ru_id: AnyStr) -> List[float]:
        summary_path = '../../../output'
        summary_file = f'Convolutional_MNIST_metrics_{ru_id}'
        metrics_dict = EarlyStopLogger.load_summary(summary_path, summary_file)
        accuracy = metrics_dict['Accuracy']
        precision = metrics_dict['Precision']
        f1_scores = []
        for acc, pres in zip(accuracy, precision):
            f1_scores.append(2.0*acc*pres/(acc + pres).float())
        return f1_scores

    @staticmethod
    def extract_eval_losses(ru_id: AnyStr) -> List[float]:
        summary_path = '../../../output'
        summary_file = f'Convolutional_MNIST_metrics_{ru_id}'
        metrics_dict = EarlyStopLogger.load_summary(summary_path, summary_file)
        return [t.item() for t in metrics_dict['Evaluation loss']]

    @staticmethod
    def create_and_train_network(lr: float, activation: nn.Module) -> NoReturn:
        from dl.training.hyper_params import HyperParams

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
                momentum=0.89,
                epochs=10,
                optim_label='adam',
                batch_size=16,
                loss_function=nn.CrossEntropyLoss(),
                drop_out=0.15,
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
            print(repr(conv_MNIST_instance))
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

