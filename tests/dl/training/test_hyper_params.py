import unittest
from torch import nn
from dl.training.hyper_params import HyperParams
from dl.block.mlp_block import MLPBlock
from dl.model.mlp_model import MLPModel
import logging
import util


class HyperParamsTest(unittest.TestCase):

    def test_init(self):
        hyper_parameters = HyperParams(
            lr=0.001,
            momentum=0.95,
            epochs=12,
            optim_label='adam',
            batch_size=8,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.2,
            train_eval_ratio=0.9)
        logging.info(repr(hyper_parameters))

    def test_optimizer(self):
        hyper_parameters = HyperParams(
            lr=0.001,
            momentum=0.95,
            epochs=12,
            optim_label='adam',
            batch_size=8,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.2,
            train_eval_ratio=0.9)
        input_block = MLPBlock.build_from_params('../../../python/input', 32, 16, nn.ReLU())
        hidden_block = MLPBlock.build_from_params('hidden', 16, 5, nn.ReLU())
        output_block = MLPBlock.build_from_params('output', 5, 5, nn.Softmax())
        ffnn_model = MLPModel('test1', [input_block, hidden_block, output_block])
        opt = hyper_parameters.optimizer(ffnn_model)
        logging.info(str(opt))


if __name__ == '__main__':
    unittest.main()
