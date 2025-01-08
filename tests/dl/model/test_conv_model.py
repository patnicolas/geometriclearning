import unittest
import torch.nn as nn
from dl.block.cnn.conv_2d_block import Conv2DBlock
from dl.block.ffnn_block import FFNNBlock
from dl.model.conv_model import ConvModel
from dl import ConvException
from dl.training.neural_training import NeuralTraining


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
                                                bias=False,
                                                drop_out=0.2)
            conv_2d_block_2 = Conv2DBlock.build(block_id='conv_2',
                                                in_channels=8,
                                                out_channels=16,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1,1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False,
                                                drop_out=0.2)
            num_classes = 10
            ffnn_block_1 = FFNNBlock.build(block_id='hidden',
                                           layer=nn.Linear(in_features=0, out_features=num_classes, bias=False),
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
                                                bias=False,
                                                drop_out=0.25)
            conv_2d_block_3 = Conv2DBlock.build(block_id='conv_3',
                                                in_channels=16,
                                                out_channels=32,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False,
                                                drop_out=0.25)
            ffnn_block_1 = FFNNBlock.build(block_id='hidden',
                                           layer=nn.Linear(in_features=0, out_features=64, bias=False),
                                           activation=nn.ReLU())
            num_classes = 10
            ffnn_block_2 = FFNNBlock.build(block_id='output',
                                           layer=nn.Linear(in_features=64, out_features=num_classes, bias=False),
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
                                                bias=False,
                                                drop_out=0.25)
            conv_2d_block_2 = Conv2DBlock.build(block_id='conv_2',
                                                in_channels=8,
                                                out_channels=16,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False,
                                                drop_out=0.25)
            conv_2d_block_3 = Conv2DBlock.build(block_id='conv_3',
                                                in_channels=16,
                                                out_channels=32,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False,
                                                drop_out=0.25)
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

    @unittest.skip('Ignore')
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
                                                bias=False,
                                                drop_out=0.25)
            conv_2d_block_2 = Conv2DBlock.build(block_id='conv_2',
                                                in_channels=8,
                                                out_channels=16,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False,
                                                drop_out=0.25)
            num_classes = 10
            ffnn_block_1 = FFNNBlock.build(block_id='hidden',
                                           layer=nn.Linear(in_features=0, out_features=num_classes, bias=False),
                                           activation=nn.Softmax(dim=0))
            conv_model = ConvModel(model_id='MNIST',
                                   input_size=(28, 28),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2],
                                   ffnn_blocks=[ffnn_block_1])
            print(repr(conv_model))
            mnist_loader = MNISTLoader(batch_size=8)
            train_loader, eval_loader = mnist_loader.loaders_from_path(root_path='../../../data/MNIST',
                                                                       exec_config=ExecConfig.default())
            net_training = ConvModelTest.create_executor()
            net_training.train(conv_model.model_id, conv_model, train_loader, eval_loader)
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_caltech101_train(self):
        from dataset.caltech101_loader import Caltech101Loader
        from dl.training.exec_config import ExecConfig

        try:
            target_size = 128
            conv_2d_block_1 = Conv2DBlock.build(block_id='conv_1',
                                                in_channels=3,
                                                out_channels=64,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False,
                                                drop_out=0.25)
            conv_2d_block_2 = Conv2DBlock.build(block_id='conv_2',
                                                in_channels=64,
                                                out_channels=128,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False,
                                                drop_out=0.25)
            conv_2d_block_3 = Conv2DBlock.build(block_id='conv_3',
                                                in_channels=128,
                                                out_channels=256,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False,
                                                drop_out=0.25)
            num_classes = 101
            ffnn_block_1 = FFNNBlock.build(block_id='hidden',
                                           layer=nn.Linear(in_features=0, out_features=512, bias=False),
                                           activation=nn.Softmax(dim=0))
            ffnn_block_2 = FFNNBlock.build(block_id='output',
                                           layer=nn.Linear(in_features=512, out_features=num_classes, bias=False),
                                           activation=nn.Softmax(dim=0))
            conv_model = ConvModel(model_id='Caltech-101',
                                   input_size=(target_size, target_size),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2, conv_2d_block_3],
                                   ffnn_blocks=[ffnn_block_1, ffnn_block_2])
            print(repr(conv_model))
            caltech101_loader = Caltech101Loader(batch_size=8, split_ratio=0.9, resize_image=128)
            train_loader, eval_loader = caltech101_loader.loaders_from_path(root_path='../../../data/caltech-101',
                                                                            exec_config=ExecConfig.default())
            net_training = ConvModelTest.create_executor()
            net_training.train(conv_model.model_id, conv_model, train_loader, eval_loader)
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @staticmethod
    def create_executor() -> NeuralTraining:
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
        metric_labels = [Metric.accuracy_label, Metric.precision_label, Metric.recall_label]
        return NeuralTraining.build(hyper_parameters, metric_labels)


if __name__ == '__main__':
    unittest.main()
