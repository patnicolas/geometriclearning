import unittest
import torch
from dl.block.mlp_block import MLPBlock
from dl.model.mlp_model import MLPModel
from dl.model.ae_model import AEModel


class AEModelTest(unittest.TestCase):
    def test_init(self):
        input_block = MLPBlock.build('input', 128, 32, torch.nn.ReLU())
        hidden_block = MLPBlock.build('hidden', 32, 16, torch.nn.ReLU())
        output_block = MLPBlock.build('output', 16, 8, torch.nn.ReLU())
        ffnn_model = MLPModel('test1', [input_block, hidden_block, output_block])

        ae_model = AEModel('Autoencoder', ffnn_model)
        print(repr(ae_model))


if __name__ == '__main__':
    unittest.main()

