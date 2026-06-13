import unittest
import torch
from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning.model.mlp.mlp_model import MLPModel
from deeplearning.model.generative.ae_model import AEModel
import logging


class AEModelTest(unittest.TestCase):
    def test_init(self):
        input_block = MLPBlock.build_from_params('input', 128, 32, torch.nn.ReLU())
        hidden_block = MLPBlock.build_from_params('hidden', 32, 16, torch.nn.ReLU())
        output_block = MLPBlock.build_from_params('output', 16, 8, torch.nn.ReLU())
        ffnn_model = MLPModel('test1', [input_block, hidden_block, output_block])

        ae_model = AEModel('Autoencoder', ffnn_model)
        logging.info(repr(ae_model))


if __name__ == '__main__':
    unittest.main()

