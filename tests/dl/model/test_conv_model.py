import unittest
import torch.nn as nn
from dl.block.conv.conv_2d_block import Conv2dBlock
from dl.block.mlp_block import MLPBlock
from dl.model.conv_model import ConvModel
from dl import ConvException
from dl.training.neural_training import NeuralTraining


class ConvModelTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_mnist_small(self):
        try:
            conv_2d_block_1 = Conv2dBlock.build_from_params(block_id='conv_1',
                                                            in_channels=1,
                                                            out_channels=8,
                                                            kernel_size=(3, 3),
                                                            stride=(1, 1),
                                                            padding=(1, 1),
                                                            batch_norm=True,
                                                            max_pooling_kernel=2,
                                                            activation=nn.ReLU(),
                                                            bias=False,
                                                            drop_out=0.2)
            conv_2d_block_2 = Conv2dBlock.build_from_params(block_id='conv_2',
                                                            in_channels=8,
                                                            out_channels=16,
                                                            kernel_size=(3, 3),
                                                            stride=(1, 1),
                                                            padding=(1, 1),
                                                            batch_norm=True,
                                                            max_pooling_kernel=2,
                                                            activation=nn.ReLU(),
                                                            bias=False,
                                                            drop_out=0.2)
            num_classes = 10
            mlp_block = MLPBlock(block_id='hidden',
                                 layer_module=nn.Linear(in_features=0, out_features=num_classes, bias=False),
                                 activation_module=nn.Softmax(dim=1))
            conv_model = ConvModel(model_id='MNIST',
                                   input_size=(28, 28),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2],
                                   mlp_blocks=[mlp_block])
            print(repr(conv_model))
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_mnist_small_2(self):
        from dl.model.conv_2d_model import Conv2dBuilder
        try:
            conv_builder = Conv2dBuilder({})
            conv_builder.set(key='model_id', value='My model')
            # Initialization for convolutional layer
            conv_builder.set(key='input_size', value=(28, 28))
            conv_builder.set(key='in_channels_list', value=[8, 16, 16, 1])
            conv_builder.set(key='is_batch_norm', value=True)
            conv_builder.set(key='kernel_size', value=(3, 3))
            conv_builder.set(key='stride', value=(1, 1))
            conv_builder.set(key='padding', value=(1, 1))
            conv_builder.set(key='activation', value=nn.ReLU())
            conv_builder.set(key='max_pool_kernel', value=2)
            conv_builder.set(key='drop_out', value=0.2)

            # Initialization for fully connected layer
            conv_builder.set(key='in_features_list', value=[0, 64, 10])
            conv_builder.set(key='output_activation', value=nn.Softmax(dim=1))

            conv_model = conv_builder.build()
            print(repr(conv_model))
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    def test_mnist_small_3(self):
        from dl.model.conv_2d_model import Conv2dBuilder
        try:
            model_attributes = {
                'model_id': 'My model',
                'input_size': (28, 28),
                'in_channels_list':[8, 16, 16, 1],
                'is_batch_norm': True,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (1, 1),
                'bias': True,
                'activation': nn.ReLU(),
                'max_pool_kernel': 2,
                'drop_out': 0.2,
                'in_features_list': [0, 64, 10],
                'output_activation': nn.Softmax(dim=1)
            }
            conv_builder = Conv2dBuilder(model_attributes)
            conv_model = conv_builder.build()
            print(repr(conv_model))
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_validation_conv(self):
        from dl.model.conv_2d_model import Conv2dBuilder

        conv_2d_block_1 = Conv2dBlock.build_from_params(block_id='conv_1',
                                                        in_channels=1,
                                                        out_channels=8,
                                                        kernel_size=(3, 3),
                                                        stride=(1, 1),
                                                        padding=(1, 1),
                                                        batch_norm=True,
                                                        max_pooling_kernel=2,
                                                        activation=nn.ReLU(),
                                                        bias=False)
        conv_2d_block_2 = Conv2dBlock.build_from_params(block_id='conv_2',
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
        conv_2d_block_3 = Conv2dBlock.build_from_params(block_id='conv_3',
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
        Conv2dBuilder.validate_conv(conv_blocks=[conv_2d_block_1, conv_2d_block_2, conv_2d_block_3],
                                    input_size=(28, 28))

    @unittest.skip('Ignore')
    def test_mnist_large(self):
        try:
            conv_2d_block_1 = Conv2dBlock.build_from_params(block_id='conv_1',
                                                            in_channels=1,
                                                            out_channels=8,
                                                            kernel_size=(3, 3),
                                                            stride=(1, 1),
                                                            padding=(1, 1),
                                                            batch_norm=True,
                                                            max_pooling_kernel=2,
                                                            activation=nn.ReLU(),
                                                            bias=False)
            conv_2d_block_2 = Conv2dBlock.build_from_params(block_id='conv_2',
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
            conv_2d_block_3 = Conv2dBlock.build_from_params(block_id='conv_3',
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
            mlp_block_1 = MLPBlock.build_from_params(block_id='hidden',
                                                     in_features=0,
                                                     out_features=64,
                                                     bias=False,
                                                     activation_module=nn.ReLU())
            num_classes = 10
            mlp_block_2 = MLPBlock.build_from_params(block_id='output',
                                                     in_features=64,
                                                     out_features=num_classes,
                                                     bias=False,
                                                     activation_module=nn.Softmax(dim=1))
            conv_model = ConvModel(model_id='MNIST',
                                   input_size=(28, 28),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2, conv_2d_block_3],
                                   mlp_blocks=[mlp_block_1, mlp_block_2])
            print(repr(conv_model))
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_builder(self):
        from dl.model.conv_2d_model import Conv2dBuilder
        conv_attributes = {
            'input_size': (28, 28),
            'in_channels_list': [3, 8, 16],
            'kernel_size': (3, 3),
            'stride': (1, 1),
            'padding': (1, 1),
            'is_batch_norm': True,
            'in_features_list': [256, 20],
            'output_activation': nn.Softmax(),
            'bias': False,
            'drop_out': 0.2
        }
        conv_2d_Builder = Conv2dBuilder(conv_attributes)
        conv_2d_model = conv_2d_Builder.build()
        print(str(conv_2d_model))

    @unittest.skip('Ignore')
    def test_transpose(self):
        try:
            conv_2d_block_1 = Conv2dBlock.build_from_params(block_id='conv_1',
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
            conv_2d_block_2 = Conv2dBlock.build_from_params(block_id='conv_2',
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
            conv_2d_block_3 = Conv2dBlock.build_from_params(block_id='conv_3',
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
                                   mlp_blocks=None)
            print(f'\nConv modules:---\n{repr(conv_model)}')
            de_conv_model = conv_model.transpose(extra=nn.Sigmoid())
            print(f'\nDe conv modules: ----\n{repr(de_conv_model)}')
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_mnist_train(self):
        from dataset.tensor.mnist_loader import MNISTLoader
        from dl.training.exec_config import ExecConfig

        try:
            conv_2d_block_1 = Conv2dBlock.build_from_params(block_id='conv_1',
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
            conv_2d_block_2 = Conv2dBlock.build_from_params(block_id='conv_2',
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
            mlp_block_1 = MLPBlock.build_from_params(block_id='hidden',
                                                     in_features=0,
                                                     out_features=num_classes,
                                                     bias=False,
                                                     activation_module=nn.Softmax(dim=0))
            conv_model = ConvModel(model_id='MNIST',
                                   input_size=(28, 28),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2],
                                   mlp_blocks=[mlp_block_1])
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
        from dataset.tensor.caltech101_loader import Caltech101Loader
        from dl.training.exec_config import ExecConfig

        try:
            target_size = 128
            conv_2d_block_1 = Conv2dBlock.build_from_params(block_id='conv_1',
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
            conv_2d_block_2 = Conv2dBlock.build_from_params(block_id='conv_2',
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
            conv_2d_block_3 = Conv2dBlock.build_from_params(block_id='conv_3',
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
            mlp_block_1 = MLPBlock.build_from_params(block_id='hidden',
                                                     in_features=0,
                                                     out_features=512,
                                                     bias=False,
                                                     activation_module=nn.ReLU())
            mlp_block_2 = MLPBlock.build_from_params(block_id='output',
                                                     in_features=512,
                                                     out_features=num_classes,
                                                     bias=False,
                                                     activation_module=nn.Softmax(dim=0))
            conv_model = ConvModel(model_id='Caltech-101',
                                   input_size=(target_size, target_size),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2, conv_2d_block_3],
                                   mlp_blocks=[mlp_block_1, mlp_block_2])
            print(repr(conv_model))
            caltech101_loader = Caltech101Loader(batch_size=8, split_ratio=0.9, resize_image=128)
            train_loader, eval_loader = caltech101_loader.loaders_from_path(root_path='../../../data/caltech-101',
                                                                            exec_config=ExecConfig.default())
            net_training = ConvModelTest.create_executor()
            print(f'Network training:\n{net_training}')
            net_training.train(conv_model.model_id, conv_model, train_loader, eval_loader)
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    @staticmethod
    def create_executor() -> NeuralTraining:
        from metric.metric_type import MetricType
        from dl.training.hyper_params import HyperParams
        from metric.built_in_metric import BuiltInMetric

        hyper_parameters = HyperParams(
            lr=0.001,
            momentum=0.95,
            epochs=8,
            optim_label='adam',
            batch_size=8,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.0,
            train_eval_ratio=0.9)

        metrics_attributes = {MetricType.Accuracy: BuiltInMetric(metric_type=MetricType.Accuracy),
                              MetricType.Precision: BuiltInMetric(metric_type=MetricType.Precision)}
        return NeuralTraining(hyper_parameters, metrics_attributes)



if __name__ == '__main__':
    unittest.main()
