import unittest
import torch
from python.dl.block.ffnnblock import FFNNBlock
from python.dl.model.ffnnmodel import FFNNModel
from python.dl.model.aemodel import AEModel


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

