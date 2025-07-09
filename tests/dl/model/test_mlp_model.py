import unittest
import torch.nn as nn
from dl.block.mlp_block import MLPBlock
from dl.model.mlp_model import MLPModel, MLPBuilder
from dl.training.neural_training import NeuralTraining
from dl import MLPException
import logging
import os
import python
from python import SKIP_REASON


class MLPModelTest(unittest.TestCase):

    def test_init_1(self):
        try:
            input_block = MLPBlock(block_id='input',
                                   layer_module=nn.Linear(in_features=8, out_features=16, bias=False),
                                   activation_module=nn.ReLU())
            hidden_block = MLPBlock(block_id='hidden',
                                    layer_module=nn.Linear(in_features=16, out_features=16, bias=False),
                                    activation_module=nn.ReLU())
            output_block = MLPBlock(block_id='output',
                                    layer_module=nn.Linear(in_features=16, out_features=1, bias=False),
                                    activation_module=nn.Softmax())
            mlp_model = MLPModel(model_id='test1',
                                 mlp_blocks=[input_block, hidden_block, output_block])
            self.assertTrue(mlp_model.get_in_features() == 8)
            self.assertTrue(mlp_model.get_out_features() == 1)
            params = list(mlp_model.parameters())
            logging.info(f'Parameters:\n{params}')
            self.assertTrue(len(params) == 3)

            mlp_builder = MLPBuilder({})
            # 'in_features_list', 'activation', 'drop_out', 'output_activation'
            mlp_builder.set(key='model_id', value='my_model')
            mlp_builder.set(key='in_features_list', value=[8, 16, 16, 1])
            mlp_builder.set(key='drop_out', value=0.5)
            mlp_builder.set(key='activation', value=nn.ReLU())
            mlp_builder.set(key='output_activation', value=nn.Softmax())
            mlp_model = mlp_builder.build()
            logging.info(str(mlp_model))
            self.assertTrue(True)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except MLPException as e:
            logging.info(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_2(self):
        try:
            input_block = MLPBlock.build_from_params(block_id='input',
                                                     in_features=8,
                                                     out_features=16,
                                                     activation_module= nn.ReLU(),
                                                     dropout_p=0.3)
            hidden_block = MLPBlock.build_from_params(block_id='hidden',
                                                      in_features=16,
                                                      out_features=16,
                                                      activation_module= nn.ReLU(),
                                                      dropout_p=0.3)
            output_block = MLPBlock.build_from_params(block_id='output',
                                                      in_features=16,
                                                      out_features=1,
                                                      activation_module=nn.Softmax())

            mlp_model = MLPModel(model_id='test1',
                                 mlp_blocks=[input_block, hidden_block, output_block])
            self.assertTrue(mlp_model.get_in_features() == 8)
            self.assertTrue(mlp_model.get_out_features() == 1)
            logging.info(repr(mlp_model))
            self.assertTrue(True)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except MLPException as e:
            logging.error(e)
            self.assertTrue(False)


    def test_builder(self):
        model_attributes = {
            'model_id': 'my_mlp',
            'in_features_list': [8, 1, 16],
            'activation': nn.ReLU(),
            'drop_out': 0.3,
            'output_activation': nn.Softmax()
        }
        mlp_builder = MLPBuilder(model_attributes)
        mlp_model = mlp_builder.build()
        logging.info(mlp_model)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_transpose(self):
        try:
            input_block = MLPBlock.build_from_params(block_id='input',
                                                     in_features=8,
                                                     out_features=16,
                                                     activation_module=nn.ReLU(),
                                                     dropout_p=0.3)
            hidden_block = MLPBlock.build_from_params(block_id='hidden',
                                                      in_features=16,
                                                      out_features=16,
                                                      activation_module=nn.ReLU(),
                                                      dropout_p=0.3)
            output_block = MLPBlock.build_from_params(block_id='output',
                                                      in_features=16,
                                                      out_features=1,
                                                      activation_module=nn.Softmax())
            mlp_model = MLPModel(model_id='test1',
                                 mlp_blocks=[input_block, hidden_block, output_block])
            mlp_model_transposed = mlp_model.transpose(output_activation=None)
            self.assertTrue(mlp_model_transposed.in_features == 1)
            self.assertTrue(mlp_model_transposed.out_features == 8)
            self.assertTrue(True)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except MLPException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_train_mnist(self):
        try:
            # Input layer
            from dataset.tensor.mnist_loader import MNISTLoader
            from dl.training.exec_config import ExecConfig

            features = [256, 128, 64]
            num_classes = 10
            mlp_input_block = MLPBlock(block_id='input',
                                       layer_module=nn.Linear(in_features=784, out_features=num_classes, bias=False),
                                       activation_module=nn.ReLU())

            # Hidden layers if any
            mlp_hidden_blocks = [MLPBlock(block_id=f'hidden_{idx + 1}',
                                          layer_module=nn.Linear(in_features=features[idx],
                                                                 out_features=features[idx + 1],
                                                                 bias=False),
                                          activation_module=nn.ReLU()) for idx in range(len(features[:-1]))]
            # Output layer
            mlp_output_block = MLPBlock(block_id='output',
                                        layer_module=nn.Linear(in_features=features[-1],
                                                               out_features=num_classes,
                                                               bias=False),
                                        activation_module=nn.Softmax(dim=1))

            # Define the model and layout for the Feed Forward Neural Network
            mlp_model = MLPModel(model_id='MNIST-ML)',
                                 mlp_blocks=[mlp_input_block] + mlp_hidden_blocks + [mlp_output_block])
            mnist_loader = MNISTLoader(batch_size=128)
            train_loader, eval_loader = mnist_loader.loaders_from_path(root_path='../../../data/MNIST',
                                                                       exec_config=ExecConfig.default())
            net_training = MLPModelTest.create_executor()
            net_training.train(mlp_model.model_id, mlp_model, train_loader, eval_loader)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except MLPException as e:
            logging.error(e)
            self.assertTrue(False)

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