import unittest
import torch.nn as nn
from dl.block.ffnn_block import FFNNBlock
from dl.model.ffnn_model import FFNNModel
from dl.training.neural_training import NeuralTraining
from dl import DLException


class FFNNModelTest(unittest.TestCase):

    @unittest.skip("Ignore")
    def test_init_1(self):
        try:
            input_block = FFNNBlock.build(block_id='input',
                                          layer=nn.Linear(in_features=8, out_features=16, bias=False),
                                          activation= nn.ReLU())
            hidden_block = FFNNBlock.build(block_id='hidden',
                                           layer=nn.Linear(in_features=16, out_features=16, bias=False),
                                           activation= nn.ReLU())
            output_block = FFNNBlock.build(block_id='output',
                                           layer=nn.Linear(in_features=16, out_features=1, bias=False),
                                           activation=nn.Softmax())
            ffnn_model = FFNNModel(model_id='test1', neural_blocks=[input_block, hidden_block, output_block])
            self.assertTrue(ffnn_model.in_features == 8)
            self.assertTrue(ffnn_model.out_features == 1)
            print(repr(ffnn_model))
            assert True
        except DLException as e:
            assert False

    @unittest.skip("Ignore")
    def test_init_2(self):
        try:
            input_block = FFNNBlock.build(block_id='input',
                                          layer=nn.Linear(in_features=8, out_features=16, bias=False),
                                          activation= nn.ReLU(),
                                          drop_out=0.3)
            hidden_block = FFNNBlock.build(block_id='hidden',
                                           layer=nn.Linear(in_features=16, out_features=16, bias=False),
                                           activation= nn.ReLU(),
                                           drop_out=0.3)
            output_block = FFNNBlock.build(block_id='output',
                                           layer=nn.Linear(in_features=16, out_features=1, bias=False),
                                           activation=nn.Softmax())
            ffnn_model = FFNNModel(model_id='test1', neural_blocks=[input_block, hidden_block, output_block])
            self.assertTrue(ffnn_model.in_features == 8)
            self.assertTrue(ffnn_model.out_features == 1)
            print(repr(ffnn_model))
            assert True
        except DLException as e:
            assert False

    def test_transpose(self):
        try:
            input_block = FFNNBlock.build(block_id='input',
                                          layer=nn.Linear(in_features=8, out_features=16, bias=False),
                                          activation=nn.ReLU(),
                                          drop_out=0.3)
            hidden_block = FFNNBlock.build(block_id='hidden',
                                           layer=nn.Linear(in_features=16, out_features=16, bias=False),
                                           activation=nn.ReLU(),
                                           drop_out=0.3)
            output_block = FFNNBlock.build(block_id='output',
                                           layer=nn.Linear(in_features=16, out_features=1, bias=False),
                                           activation=nn.Softmax())
            ffnn_model = FFNNModel(model_id='test1', neural_blocks=[input_block, hidden_block, output_block])
            ffnn_model_transposed = ffnn_model.transpose(extra=None)
            self.assertTrue(ffnn_model_transposed.in_features == 1)
            self.assertTrue(ffnn_model_transposed.out_features == 8)
            assert True
        except DLException as e:
            assert False

    def test_train_mnist(self):
        # Input layer
        from dl.training.hyper_params import HyperParams
        from dataset.mnist_loader import MNISTLoader
        from dl.training.exec_config import ExecConfig

        features = [256, 128, 64]
        root_path = '../../../../data/MNIST'
        hyper_parameters = HyperParams(
            lr=0.0005,
            momentum=0.95,
            epochs=24,
            optim_label='adam',
            batch_size=16,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.20,
            train_eval_ratio=0.9,
            weight_initialization=False)

        num_classes = 10
        ffnn_input_block = FFNNBlock.build(block_id='input',
                                           layer=nn.Linear(in_features=784, out_features=num_classes, bias=False),
                                           activation=nn.ReLU())

        # Hidden layers if any
        ffnn_hidden_blocks = [FFNNBlock.build(block_id=f'hidden_{idx + 1}',
                                              layer=nn.Linear(in_features=features[idx],
                                                              out_features=features[idx + 1],
                                                              bias=False),
                                              activation=nn.ReLU()) for idx in range(len(features[:-1]))]
        # Output layer
        ffnn_output_block = FFNNBlock.build(block_id='output',
                                            layer=nn.Linear(in_features=features[-1],
                                                            out_features=num_classes,
                                                            bias=False),
                                            activation=nn.Softmax(dim=1))

        # Define the model and layout for the Feed Forward Neural Network
        ffnn_model = FFNNModel(model_id='MNIST-FFNN',
                               neural_blocks=[ffnn_input_block] + ffnn_hidden_blocks + [ffnn_output_block])
        mnist_loader = MNISTLoader()
        train_loader, eval_loader = mnist_loader.loaders_from_path(root_path='../../../data/MNIST',
                                                                   exec_config=ExecConfig.default())
        net_training = FFNNModelTest.create_executor()
        net_training.train(ffnn_model.model_id, ffnn_model, train_loader, eval_loader)

    @staticmethod
    def create_executor() -> NeuralTraining:
        from dl.training.hyper_params import HyperParams
        from metric.metric import Metric

        hyper_parameters = HyperParams(
            lr=0.001,
            momentum=0.95,
            epochs=8,
            optim_label='adam',
            batch_size=8,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.0,
            train_eval_ratio=0.9)
        metric_labels = [Metric.accuracy_label, Metric.precision_label, Metric.recall_label]
        return NeuralTraining.build(hyper_parameters, metric_labels)


if __name__ == '__main__':
    unittest.main()