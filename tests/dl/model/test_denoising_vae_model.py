import unittest
import torch
from dl.block.mlp_block import MLPBlock
from dl.model.mlp_model import MLPModel
from dl.model.denoising_vae_model import DenoisingVAEModel
import logging
import python


class DenoisingVAEModelTest(unittest.TestCase):
    def test_init(self):
        input_block = MLPBlock.build_from_params('in', 128, 32, torch.nn.ReLU())
        hidden_block = MLPBlock.build_from_params('hid1', 32, 10, torch.nn.ReLU())
        ffnn_model = MLPModel('encoder', [input_block, hidden_block])
        latent_size = 6
        vae_model = DenoisingVAEModel('vae_test', ffnn_model, latent_size)
        logging.info(f'\nDenoising variational auto-encoder:\n{repr(vae_model)}')
        modules = list(vae_model.modules_seq.modules())
        self.assertTrue(len(modules) == 12)


if __name__ == '__main__':
    unittest.main()

