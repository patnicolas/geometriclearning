import unittest
import torch.nn as nn
from dl.block.conv_2d_block import Conv2DBlock
from dl.block.ffnn_block import FFNNBlock
from dl.model.conv_model import ConvModel
from dl import ConvException
from dl.training.dl_training import DLTraining


class ConvModelTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_mnist_small(self):
        try:
            conv_2d_block_1 = Conv2DBlock.build(block_id='conv_1',
                                                in_channels=1,
                                                out_channels=8,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1,1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            conv_2d_block_2 = Conv2DBlock.build(block_id='conv_2',
                                                in_channels=8,
                                                out_channels=16,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1,1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            num_classes = 10
            ffnn_block_1 = FFNNBlock.build(block_id='hidden',
                                           in_features=0,
                                           out_features=num_classes,
                                           activation=nn.Softmax(dim=1))
            conv_model = ConvModel(model_id='MNIST',
                                   input_size=(28, 28),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2],
                                   ffnn_blocks=[ffnn_block_1])
            print(repr(conv_model))
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_mnist_large(self):
        try:
            conv_2d_block_1 = Conv2DBlock.build(block_id='conv_1',
                                                in_channels=1,
                                                out_channels=8,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            conv_2d_block_2 = Conv2DBlock.build(block_id='conv_2',
                                                in_channels=8,
                                                out_channels=16,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            conv_2d_block_3 = Conv2DBlock.build(block_id='conv_3',
                                                in_channels=16,
                                                out_channels=32,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            ffnn_block_1 = FFNNBlock.build(block_id='hidden',
                                           in_features=0,
                                           out_features=64,
                                           activation=nn.ReLU())
            num_classes = 10
            ffnn_block_2 = FFNNBlock.build(block_id='output',
                                           in_features=64,
                                           out_features=num_classes,
                                           activation=nn.Softmax(dim=1))
            conv_model = ConvModel(model_id='MNIST',
                                   input_size=(28, 28),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2, conv_2d_block_3],
                                   ffnn_blocks=[ffnn_block_1, ffnn_block_2])
            print(repr(conv_model))
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_transpose(self):
        try:
            conv_2d_block_1 = Conv2DBlock.build(block_id='conv_1',
                                                in_channels=1,
                                                out_channels=8,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            conv_2d_block_2 = Conv2DBlock.build(block_id='conv_2',
                                                in_channels=8,
                                                out_channels=16,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            conv_2d_block_3 = Conv2DBlock.build(block_id='conv_3',
                                                in_channels=16,
                                                out_channels=32,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            conv_model = ConvModel(model_id='MNIST',
                                   input_size=(28, 28),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2, conv_2d_block_3],
                                   ffnn_blocks=None)
            print(f'\nConv modules:---\n{repr(conv_model)}')
            de_conv_model = conv_model.transpose(extra=nn.Sigmoid())
            print(f'\nDe conv modules: ----\n{repr(de_conv_model)}')
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    def test_mnist_train(self):
        from dataset.mnist_loader import MNISTLoader
        from dl.training.exec_config import ExecConfig

        try:
            conv_2d_block_1 = Conv2DBlock.build(block_id='conv_1',
                                                in_channels=3,
                                                out_channels=8,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            conv_2d_block_2 = Conv2DBlock.build(block_id='conv_2',
                                                in_channels=8,
                                                out_channels=16,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
            num_classes = 10
            ffnn_block_1 = FFNNBlock.build(block_id='hidden',
                                           in_features=0,
                                           out_features=num_classes,
                                           activation=nn.Softmax(dim=0))
            conv_model = ConvModel(model_id='MNIST',
                                   input_size=(28, 28),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2],
                                   ffnn_blocks=[ffnn_block_1])
            print(repr(conv_model))
            mnist_loader = MNISTLoader()
            train_loader, eval_loader = mnist_loader.loaders_from_path(root_path='../../../data/MNIST',
                                                                       exec_config=ExecConfig.default())
            net_training = ConvModelTest.create_executor()
            net_training.train(conv_model.model_id, conv_model, train_loader, eval_loader)
            self.assertTrue(True)

        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @staticmethod
    def create_executor() -> DLTraining:
        from dl.training.hyper_params import HyperParams
        from metric.metric import Metric

        hyper_parameters = HyperParams(
            lr=0.001,
            momentum=0.95,
            epochs=8,
            optim_label='adam',
            batch_size=8,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.0,
            train_eval_ratio=0.9)
        metric_labels = [ Metric.accuracy_label, Metric.precision_label, Metric.recall_label]
        return DLTraining.build(hyper_parameters, metric_labels)


if __name__ == '__main__':
    unittest.main()
