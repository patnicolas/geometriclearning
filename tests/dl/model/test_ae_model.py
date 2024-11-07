import unittest
import torch
from dl.block.ffnn_block import FFNNBlock
from dl.model.ffnn_model import FFNNModel
from dl.model.ae_model import AEModel


class AEModelTest(unittest.TestCase):
    def test_init(self):
        input_block = FFNNBlock.build('input', 128, 32, torch.nn.ReLU())
        hidden_block = FFNNBlock.build('hidden', 32, 16, torch.nn.ReLU())
        output_block = FFNNBlock.build('output', 16, 8, torch.nn.ReLU())
        ffnn_model = FFNNModel('test1', [input_block, hidden_block, output_block])

        ae_model = AEModel('Autoencoder', ffnn_model)
        print(repr(ae_model))


if __name__ == '__main__':
    unittest.main()

