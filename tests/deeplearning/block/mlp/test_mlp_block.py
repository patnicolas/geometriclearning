import unittest
import logging
from torch import nn
from deeplearning.block.mlp.mlp_block import MLPBlock
from deeplearning import MLPException
import os
from python import SKIP_REASON


class MLPBlockTest(unittest.TestCase):

    # @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_1(self):
        try:
            mlp_block = MLPBlock(block_id='id1',
                                 layer_module=nn.Linear(12, 6),
                                 activation_module=nn.ReLU(),
                                 dropout_module=nn.Dropout(0.4))
            mlp_block.init_weights()
            params = mlp_block.parameters()
            params_list = list(params)
            self.assertTrue(len(params_list) == 2)
            self.assertTrue(mlp_block.get_in_features() == 12)
            self.assertTrue(mlp_block.get_out_features() == 6)

            # logging.info(str(mlp_block))
        except MLPException as e:
            logging.info(str(e))
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
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
        except MLPException as e:
            logging.info(str(e))
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
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

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
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
        except MLPException as e:
            logging.info(str(e))
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
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
        except MLPException as e:
            logging.info(str(e))
            self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()