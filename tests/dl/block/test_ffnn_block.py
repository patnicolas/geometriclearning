import unittest
from torch import nn
from dl.block.ffnn_block import FFNNBlock
from dl import DLException


class FFNNBlockTest(unittest.TestCase):

    def test_init(self):
        in_features = 24
        out_features = 12
        try:
            linear = nn.Linear(in_features, out_features, False)
            activation = nn.ReLU()
            ffnn_block = FFNNBlock('id1', linear, activation)
            self.assertTrue(ffnn_block.in_features == in_features)
            self.assertTrue(ffnn_block.out_features == out_features)
            print(repr(ffnn_block))
            self.assertTrue(True)
        except DLException as e:
            print(str(e))
            self.assertTrue(False)


    def test_init_cls(self):
        in_features = 24
        out_features = 12
        activation = nn.ReLU()
        ffnn_block = FFNNBlock.build('id', in_features, out_features, activation)
        print(repr(ffnn_block))

    def test_invert(self):
        in_features = 24
        out_features = 12
        activation = nn.ReLU()
        ffnn_block = FFNNBlock.build('id', in_features, out_features, activation, 0.2)
        print(f'Model: {repr(ffnn_block)}')
        inverted_ffnn_block = ffnn_block.invert()
        print(f'Inverted:\n{repr(inverted_ffnn_block)}')


if __name__ == '__main__':
    unittest.main()