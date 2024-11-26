import unittest
import torch
import torch.nn as nn
from dl.block.ffnn_block import FFNNBlock
from dl.model.ffnn_model import FFNNModel
from dl.model.denoising_vae_model import DenoisingVAEModel


class DenoisingVAEModelTest(unittest.TestCase):
    def test_init(self):
        input_block = FFNNBlock.build('in', 128, 32, torch.nn.ReLU())
        hidden_block = FFNNBlock.build('hid1', 32, 10, torch.nn.ReLU())
        ffnn_model = FFNNModel('encoder', [input_block, hidden_block])
        latent_size = 6
        vae_model = DenoisingVAEModel('vae_test', ffnn_model, latent_size)
        print(f'\nDenoising variational auto-encoder:\n{repr(vae_model)}')
        modules = list(vae_model.model.modules())
        self.assertTrue(len(modules) == 12)


if __name__ == '__main__':
    unittest.main()

