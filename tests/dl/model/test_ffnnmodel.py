import unittest
import torch
from dl.block.ffnnblock import FFNNBlock
from dl.model.ffnnmodel import FFNNModel


class FFNNModelTest(unittest.TestCase):

    def test_init(self):
        input_block = FFNNBlock.build('input', 32, 16, torch.nn.ReLU())
        hidden_block = FFNNBlock.build('hidden', 16, 5, torch.nn.ReLU())
        output_block = FFNNBlock.build('output', 5, 5, torch.nn.Softmax())
        ffnn_model = FFNNModel('test1', [input_block, hidden_block, output_block])
        print(repr(ffnn_model))

    def test_invert(self):
        input_block = FFNNBlock.build('input', 32, 16, torch.nn.ReLU())
        hidden_block = FFNNBlock.build('hidden', 16, 5, torch.nn.ReLU())
        output_block = FFNNBlock.build('output', 5, 5, torch.nn.ReLU())
        ffnn_model = FFNNModel('test1', [input_block, hidden_block, output_block])
        print(f'Original {repr(ffnn_model)}')
        inverted_ffnn_model = ffnn_model.invert()
        print(f'Inverted: {repr(inverted_ffnn_model)}')


if __name__ == '__main__':
    unittest.main()