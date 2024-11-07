import unittest
import torch
from dl.block.ffnn_block import FFNNBlock
from dl.model.ffnn_model import FFNNModel
from dl.model.vae_model import VAEModel


class VAEModelTest(unittest.TestCase):
    def test_init(self):
        input_block = FFNNBlock.build('in', 128, 32, torch.nn.ReLU())
        hidden_block = FFNNBlock.build('hid1', 32, 10, torch.nn.ReLU())
        ffnn_model = FFNNModel('encoder', [input_block, hidden_block])
        latent_size = 6
        vae_model = VAEModel('vae_test', ffnn_model, latent_size)
        print(repr(vae_model))


if __name__ == '__main__':
    unittest.main()

