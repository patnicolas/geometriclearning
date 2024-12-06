import unittest
import torch
import torch.nn as nn
from dl.block.ffnn_block import FFNNBlock
from dl.model.ffnn_model import FFNNModel
from dl.model.conv_model import ConvModel
from dl.model.vae_model import VAEModel
from dl.block.conv_block import ConvBlock
from dl import ConvException, VAEException
from dl.block.builder.conv2d_block_builder import Conv2DBlockBuilder


class VAEModelTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init_1(self):
        input_block = FFNNBlock.build('in', 128, 32, torch.nn.ReLU())
        hidden_block = FFNNBlock.build('hid1', 32, 10, torch.nn.ReLU())
        ffnn_model = FFNNModel('encoder', [input_block, hidden_block])
        latent_size = 6
        vae_model = VAEModel(model_id='vae_test', encoder=ffnn_model, latent_size=latent_size, noise_func =None)
        print(repr(vae_model))

    @unittest.skip('Ignore')
    def test_init_2(self):
        conv_block_builder = Conv2DBlockBuilder(in_channels=1,
                                                out_channels=32,
                                                input_size=(28, 28),
                                                kernel_size=(3, 3),
                                                stride=(1, 1,),
                                                padding=(0, 0),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
        conv1_block = ConvBlock(_id='Conv1', conv_block_builder=conv_block_builder)
        conv_block_builder = Conv2DBlockBuilder(in_channels=32,
                                                out_channels=64,
                                                input_size=(28, 28),
                                                kernel_size=(3, 3),
                                                stride=(1, 1,),
                                                padding=(0, 0),
                                                batch_norm=True,
                                                max_pooling_kernel=2,
                                                activation=nn.ReLU(),
                                                bias=False)
        conv2_block = ConvBlock(_id='Conv2', conv_block_builder=conv_block_builder)
        ffnn1_block = FFNNBlock.build('ffnn1', 64, 32, nn.ReLU())
        ffnn2_block = FFNNBlock.build('ffnn2', 32, 1, nn.Sigmoid())

        encoder = ConvModel(model_id='MNIST', conv_blocks= [conv1_block, conv2_block],  ffnn_blocks=[ffnn1_block, ffnn2_block])
        vae_mnist_model = VAEModel(model_id='VAE_MNIST', encoder=encoder, latent_size=16)
        print(f'\nVAE MNIST config:\n{repr(vae_mnist_model)}')

    def test_init3(self):
        encoder = VAEModelTest.create_mnist_conv()
        if encoder is not None:
            latent_size = 64
            decoder_out_activation = nn.Sigmoid()
            vae_model = VAEModel('VAE - Mnist', encoder, latent_size, decoder_out_activation)
            print(str(vae_model))
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    @staticmethod
    def create_mnist_conv() -> ConvModel | None:
        try:
            conv_2d_block_builder = Conv2DBlockBuilder(
                in_channels=1,
                out_channels=32,
                input_size=(28, 28),
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                batch_norm=True,
                max_pooling_kernel=-1,
                activation=nn.ReLU(),
                bias=False)
            conv_block_1 = ConvBlock(_id='Conv1', conv_block_builder=conv_2d_block_builder)
            conv_2d_block_builder = Conv2DBlockBuilder(
                in_channels=32,
                out_channels=64,
                input_size=(28, 28),
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                batch_norm=True,
                max_pooling_kernel=-1,
                activation=nn.ReLU(),
                bias=False)
            conv_block_2 = ConvBlock(_id='Conv2', conv_block_builder=conv_2d_block_builder)
            return ConvModel(model_id ='conv_MNIST_model', conv_blocks=[conv_block_1, conv_block_2])
        except ConvException as e:
            print(str(e))
            return None
        except VAEException as e:
            print(str(e))
            return None

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

