
import unittest
from torch import nn
from python.dl.hyperparams import HyperParams
from python.dl.block.ffnnblock import FFNNBlock
from python.dl.model.ffnnmodel import FFNNModel


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
        print(repr(hyper_parameters))

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
        input_block = FFNNBlock.build('input', 32, 16, nn.ReLU())
        hidden_block = FFNNBlock.build('hidden', 16, 5, nn.ReLU())
        output_block = FFNNBlock.build('output', 5, 5, nn.Softmax())
        ffnn_model = FFNNModel('test1', [input_block, hidden_block, output_block])
        opt = hyper_parameters.optimizer(ffnn_model)
        print(str(opt))


if __name__ == '__main__':
    unittest.main()
