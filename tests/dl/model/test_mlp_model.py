import unittest
import torch.nn as nn
from dl.block.mlp_block import MLPBlock
from dl.model.mlp_model import MLPModel, MLPBuilder
from dl.training.neural_training import NeuralTraining
from dl import DLException


class MLPModelTest(unittest.TestCase):

    @unittest.skip("Ignore")
    def test_init_1(self):
        try:

            input_block = MLPBlock.build(block_id='input',
                                         layer=nn.Linear(in_features=8, out_features=16, bias=False),
                                         activation= nn.ReLU())
            hidden_block = MLPBlock.build(block_id='hidden',
                                          layer=nn.Linear(in_features=16, out_features=16, bias=False),
                                          activation= nn.ReLU())
            output_block = MLPBlock.build(block_id='output',
                                          layer=nn.Linear(in_features=16, out_features=1, bias=False),
                                          activation=nn.Softmax())
            mlp_model = MLPModel(model_id='test1',
                                  neural_blocks=[input_block, hidden_block, output_block])
            self.assertTrue(mlp_model.in_features == 8)
            self.assertTrue(mlp_model.out_features == 1)
            print(repr(mlp_model))
            assert True
        except DLException as e:
            assert False

    @unittest.skip("Ignore")
    def test_init_2(self):
        try:
            input_block = MLPBlock.build(block_id='input',
                                         layer=nn.Linear(in_features=8, out_features=16, bias=False),
                                         activation= nn.ReLU(),
                                         drop_out=0.3)
            hidden_block = MLPBlock.build(block_id='hidden',
                                          layer=nn.Linear(in_features=16, out_features=16, bias=False),
                                          activation= nn.ReLU(),
                                          drop_out=0.3)
            output_block = MLPBlock.build(block_id='output',
                                          layer=nn.Linear(in_features=16, out_features=1, bias=False),
                                          activation=nn.Softmax())
            mlp_model = MLPModel(model_id='test1',
                                  neural_blocks=[input_block, hidden_block, output_block])
            self.assertTrue(mlp_model.in_features == 8)
            self.assertTrue(mlp_model.out_features == 1)
            print(repr(mlp_model))
            assert True
        except DLException as e:
            assert False

    def test_builder(self):
        mlp_builder = (MLPBuilder('test').set('in_channels', [8, 16, 16])
                        .set('activation', nn.ReLU())
                        .set('drop_out', 0.3)
                        .set('output_activation', nn.Softmax()))
        mlp_model = mlp_builder.build()
        print(mlp_model)



    def test_transpose(self):
        try:
            input_block = MLPBlock.build(block_id='input',
                                         layer=nn.Linear(in_features=8, out_features=16, bias=False),
                                         activation=nn.ReLU(),
                                         drop_out=0.3)
            hidden_block = MLPBlock.build(block_id='hidden',
                                          layer=nn.Linear(in_features=16, out_features=16, bias=False),
                                          activation=nn.ReLU(),
                                          drop_out=0.3)
            output_block = MLPBlock.build(block_id='output',
                                          layer=nn.Linear(in_features=16, out_features=1, bias=False),
                                          activation=nn.Softmax())
            mlp_model = MLPModel(model_id='test1',
                                  neural_blocks=[input_block, hidden_block, output_block])
            mlp_model_transposed = mlp_model.transpose(output_activation=None)
            self.assertTrue(mlp_model_transposed.in_features == 1)
            self.assertTrue(mlp_model_transposed.out_features == 8)
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
            train_eval_ratio=0.9)

        num_classes = 10
        mlp_input_block = MLPBlock.build(block_id='input',
                                          layer=nn.Linear(in_features=784, out_features=num_classes, bias=False),
                                          activation=nn.ReLU())

        # Hidden layers if any
        mlp_hidden_blocks = [MLPBlock.build(block_id=f'hidden_{idx + 1}',
                                             layer=nn.Linear(in_features=features[idx],
                                                              out_features=features[idx + 1],
                                                              bias=False),
                                             activation=nn.ReLU()) for idx in range(len(features[:-1]))]
        # Output layer
        mlp_output_block = MLPBlock.build(block_id='output',
                                           layer=nn.Linear(in_features=features[-1],
                                                            out_features=num_classes,
                                                            bias=False),
                                           activation=nn.Softmax(dim=1))

        # Define the model and layout for the Feed Forward Neural Network
        mlp_model = MLPModel(model_id='MNIST-FFNN',
                              neural_blocks=[mlp_input_block] + mlp_hidden_blocks + [mlp_output_block])
        mnist_loader = MNISTLoader()
        train_loader, eval_loader = mnist_loader.loaders_from_path(root_path='../../../data/MNIST',
                                                                   exec_config=ExecConfig.default())
        net_training = MLPModelTest.create_executor()
        net_training.train(mlp_model.model_id, mlp_model, train_loader, eval_loader)

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