import unittest
import torch
from dl.block.ffnn_block import FFNNBlock
from dl.model.ffnn_model import FFNNModel
from dl.exception.dl_exception import DLException


class FFNNModelTest(unittest.TestCase):

    def test_init(self):
        in_features = 32
        out_features = 5
        try:
            input_block = FFNNBlock.build('input', in_features, 16, torch.nn.ReLU())
            hidden_block = FFNNBlock.build('hidden', 16, 5, torch.nn.ReLU())
            output_block = FFNNBlock.build('output', 5, out_features, torch.nn.Softmax())
            ffnn_model = FFNNModel('test1', [input_block, hidden_block, output_block])
            self.assertTrue(ffnn_model.in_features == in_features)
            self.assertTrue(ffnn_model.out_features == out_features)
            print(repr(ffnn_model))
            assert True
        except DLException as e:
            assert False

    def test_invert(self):
        in_features = 32
        out_features = 5
        input_block = FFNNBlock.build('input', in_features, 16, torch.nn.ReLU())
        hidden_block = FFNNBlock.build('hidden', 16, 5, torch.nn.ReLU())
        output_block = FFNNBlock.build('output', 5, out_features, torch.nn.ReLU())
        ffnn_model = FFNNModel('test1', [input_block, hidden_block, output_block])
        print(f'Original {repr(ffnn_model)}')
        inverted_ffnn_model = ffnn_model.invert()
        self.assertTrue(inverted_ffnn_model.in_features == out_features)
        self.assertTrue(inverted_ffnn_model.out_features == in_features)
        print(f'Inverted: {repr(inverted_ffnn_model)}')


if __name__ == '__main__':
    unittest.main()