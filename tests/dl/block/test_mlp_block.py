import unittest
import logging
from torch import nn
from dl.block.mlp_block import MLPBlock
from dl import DLException


class MLPBlockTest(unittest.TestCase):

    def test_init(self):
        data = (3.0, 5.2, 6.7, 0.6, 0.8)
        a, _, b, *_ = data
        print(b)

    @unittest.skip('Ignore')
    def test_init_1(self):
        try:
            mlp_block = MLPBlock(block_id='id1',
                                 layer_module=nn.Linear(12, 6),
                                 activation_module=nn.ReLU(),
                                 dropout_module=nn.Dropout(0.4))
            params = mlp_block.parameters()
            params_list = list(params)
            self.assertTrue(len(params_list) == 2)
            self.assertTrue(mlp_block.get_in_features() == 12)
            self.assertTrue(mlp_block.get_out_features() == 6)

            # logging.info(str(mlp_block))
        except DLException as e:
            logging.info(str(e))
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_init_2(self):
        in_features = 12
        out_features = 24
        try:
            mlp_block = MLPBlock.build_from_params(block_id='id1',
                                                   in_features=12,
                                                   out_features=24,
                                                   activation_module=nn.ReLU(),
                                                   dropout_p=0.3)
            self.assertTrue(mlp_block.get_in_features() == in_features)
            self.assertTrue(mlp_block.get_out_features() == out_features)
            logging.info(str(mlp_block))
            self.assertTrue(True)
        except DLException as e:
            logging.info(str(e))
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_init_3(self):
        in_features = 12
        out_features = 24

        block_attributes = {
            'block_id': 'MyMLP',
            'in_features': in_features,
            'out_features': out_features,
            'activation': nn.ReLU(),
            'dropout': 0.3
        }
        mlp_block = MLPBlock.build(block_attributes)
        logging.info(str(mlp_block))
        self.assertTrue(mlp_block.get_in_features() == in_features)
        self.assertTrue(mlp_block.get_out_features() == out_features)

    @unittest.skip('Ignore')
    def test_transpose_1(self):
        try:
            linear_layer = nn.Linear(in_features=12, out_features=24, bias=False)
            mlp_block = MLPBlock(block_id='id1',
                                 layer_module=linear_layer,
                                 activation_module=nn.ReLU(),
                                 dropout_module=nn.Dropout(0.4))
            logging.info(str(mlp_block))
            transposed = mlp_block.transpose()
            logging.info(str(transposed))
            self.assertTrue(transposed.get_in_features() == 24)
            self.assertTrue(transposed.get_out_features() == 12)
        except DLException as e:
            logging.info(str(e))
            self.assertTrue(False)

    @unittest.skip('Ignore')
    def test_transpose_2(self):
        try:
            linear_layer = nn.Linear(in_features=12, out_features=24, bias=False)
            mlp_block = MLPBlock(block_id='id1',
                                 layer_module=linear_layer,
                                 activation_module=nn.ReLU(),
                                 dropout_module=nn.Dropout(0.4))
            logging.info(repr(mlp_block))
            transposed = mlp_block.transpose(activation_update=nn.Sigmoid())
            logging.info(f'\nTransposed:\n{str(transposed)}\nwith new activation: {str(transposed.activation_module)}')
            # self.assertTrue(transposed.activation == [Sigmoid()])
        except DLException as e:
            logging.info(str(e))
            self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()