import unittest
from torch import nn
from dl.block.ffnn_block import FFNNBlock
from dl import DLException


class FFNNBlockTest(unittest.TestCase):

    def test_init_1(self):
        in_features = 12
        out_features = 24
        try:
            ffnn_block = FFNNBlock.build(block_id='id1',
                                         layer=nn.Linear(in_features=12, out_features=24, bias=False))
            self.assertTrue(ffnn_block.in_features == in_features)
            self.assertTrue(ffnn_block.out_features == out_features)
            print(repr(ffnn_block))
        except DLException as e:
            print(str(e))
            self.assertTrue(False)

    def test_init_2(self):
        in_features = 12
        out_features = 24
        try:
            ffnn_block = FFNNBlock.build(block_id='id1',
                                         layer=nn.Linear(in_features=12, out_features=24, bias=False),
                                         activation=nn.ReLU(),
                                         drop_out=0.3)
            self.assertTrue(ffnn_block.in_features == in_features)
            self.assertTrue(ffnn_block.out_features == out_features)
            print(repr(ffnn_block))
            self.assertTrue(True)
        except DLException as e:
            print(str(e))
            self.assertTrue(False)

    def test_transpose_1(self):
        try:
            ffnn_block = FFNNBlock.build(block_id='id1',
                                         layer=nn.Linear(in_features=12, out_features=24, bias=False),
                                         activation=nn.ReLU(),
                                         drop_out=0.3)
            print(repr(ffnn_block))
            transposed = ffnn_block.transpose()
            print(str(transposed))
            self.assertTrue(transposed.in_features == 24)
            self.assertTrue(transposed.out_features == 12)
        except DLException as e:
            print(str(e))
            self.assertTrue(False)

    def test_transpose_2(self):
        try:
            ffnn_block = FFNNBlock.build(block_id='id1',
                                         layer=nn.Linear(in_features=12, out_features=24, bias=False),
                                         activation=nn.ReLU(),
                                         drop_out=0.3)
            print(repr(ffnn_block))
            transposed = ffnn_block.transpose(activation_update=nn.Sigmoid())
            print(f'\nTransposed:\n{str(transposed)}\nwith new activation: {str(transposed.activation)}')
            # self.assertTrue(transposed.activation == [Sigmoid()])
        except DLException as e:
            print(str(e))
            self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()