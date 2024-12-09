import unittest
import torch.nn as nn

from dl.block.conv_2d_block import Conv2DBlock
from dl.block.ffnn_block import FFNNBlock
from dl.model.ffnn_model import FFNNModel
from dl.model.conv_model import ConvModel
from dl.model.vae_model import VAEModel
from dl import VAEException, ConvException
from dl.training.vae_training import VAETraining


class VAEModelTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init_1(self):
        try:
            input_block = FFNNBlock.build(block_id='Input',
                                          layer=nn.Linear(in_features=128, out_features=64, bias=False),
                                          activation=nn.ReLU())
            hidden_block = FFNNBlock.build(block_id='hidden',
                                           layer=nn.Linear(in_features=64, out_features=32, bias=False),
                                           activation=nn.ReLU())
            output_block = FFNNBlock.build(block_id='output',
                                           layer=nn.Linear(in_features=32, out_features=10, bias=False),
                                           activation=nn.Sigmoid())
            ffnn_model = FFNNModel(model_id='encoder', neural_blocks=[input_block, hidden_block, output_block ])
            print(str(ffnn_model))
            latent_size = 6
            vae_model = VAEModel(model_id='vae_ffnn', encoder=ffnn_model, latent_size=latent_size, noise_func =None)
            print(str(vae_model))
            self.assertTrue(True)
        except VAEException as e:
            print(f'ERROR: {str(e)}')
            self.assertTrue(True)

    @unittest.skip('Ignore')
    def test_init_2(self):
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
            conv_model = ConvModel(model_id='MNIST',
                                   input_size=(28, 28),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2],
                                   ffnn_blocks=None)
            print(repr(conv_model))
            latent_size = 6
            vae_model = VAEModel(model_id='vae_ffnn', encoder=conv_model, latent_size=latent_size, noise_func=None)
            print(repr(vae_model))
        except VAEException as e:
            print(f'ERROR: {str(e)}')
            self.assertTrue(True)

    def test_mnist_train(self):
        from dataset.mnist_loader import MNISTLoader
        from dl.training.exec_config import ExecConfig

        try:
            conv_2d_block_1 = Conv2DBlock.build(block_id='conv_1',
                                                in_channels=3,
                                                out_channels=8,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(0, 0),
                                                batch_norm=True,
                                                max_pooling_kernel=-1,
                                                activation=nn.ReLU(),
                                                bias=False,
                                                drop_out=0.25)
            conv_2d_block_2 = Conv2DBlock.build(block_id='conv_2',
                                                in_channels=8,
                                                out_channels=16,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(0, 0),
                                                batch_norm=True,
                                                max_pooling_kernel=-1,
                                                activation=nn.ReLU(),
                                                bias=False,
                                                drop_out=0.25)
            conv_model = ConvModel(model_id='MNIST',
                                   input_size=(28, 28),
                                   conv_blocks=[conv_2d_block_1, conv_2d_block_2],
                                   ffnn_blocks=None)
            vae_model = VAEModel(model_id='VAE_MNIST',
                                 encoder=conv_model,
                                 latent_size=16,
                                 decoder_out_activation=nn.Sigmoid())
            print(repr(vae_model))
            mnist_loader = MNISTLoader(batch_size=8)
            train_loader, eval_loader = mnist_loader.loaders_from_path(root_path='../../../data/MNIST',
                                                                       exec_config=ExecConfig.default())
            net_training = VAEModelTest.create_executor()
            net_training.train(vae_model.model_id, vae_model, train_loader, eval_loader)
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)
        except VAEException as e:
            print(str(e))
            self.assertTrue(False)

    @staticmethod
    def create_executor() -> VAETraining:
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
        return VAETraining.build(hyper_parameters, metric_labels)

"""

    def test_train(self):
        input_size = 784
        features = [256, 128]
        root_path = '../../../../data/MNIST'

        hyper_parameters = HyperParams(
            lr=0.0004,
            momentum=0.97,
            epochs=860,
            optim_label='adam',
            batch_size=32,
            loss_function=nn.MSELoss(),
            drop_out=0.2,
            train_eval_ratio=0.9,
            normal_weight_initialization=False)
        try:
            ffnn_mnist = FfnnMnist(input_size, features)
            vae_mnist = VAEMNIST(ffnn_mnist, latent_size=12)
            print(repr(vae_mnist))
            vae_mnist.do_train(root_path, hyper_parameters)
        except Exception as e:
            print(str(e))

"""

if __name__ == '__main__':
    unittest.main()

