import unittest
import torch
import torch.nn as nn
from dl.block.ffnn_block import FFNNBlock
from dl.model.ffnn_model import FFNNModel
from dl.model.conv_model import ConvModel
from dl.model.vae_model import VAEModel
from dl.block.conv_block import ConvBlock
from dl.block.builder.conv2d_block_builder import Conv2DBlockBuilder


class VAEModelTest(unittest.TestCase):

    def test_init_1(self):
        input_block = FFNNBlock.build('in', 128, 32, torch.nn.ReLU())
        hidden_block = FFNNBlock.build('hid1', 32, 10, torch.nn.ReLU())
        ffnn_model = FFNNModel('encoder', [input_block, hidden_block])
        latent_size = 6
        vae_model = VAEModel(model_id='vae_test', encoder=ffnn_model, latent_size=latent_size, noise_func =None)
        print(repr(vae_model))

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


if __name__ == '__main__':
    unittest.main()

