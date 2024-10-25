import unittest
from dl.model.custom.ffnn_mnist import FFNNMNIST
from dl.training.hyperparams import HyperParams
import torch.nn as nn

class FFNNMISTTest(unittest.TestCase):

    def test_init(self):
        features = [256, 128]
        input_size = 784
        ffnn_mnist = FFNNMNIST(input_size, features)
        print(repr(ffnn_mnist))
        self.assertTrue(True)

    def test_train(self):
        input_size = 784
        features = [256, 128]
        root_path = '../../../../data/MNIST'

        hyper_parameters = HyperParams(
            lr=0.00008,
            momentum=0.95,
            epochs=42,
            optim_label='adam',
            batch_size=32,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.2,
            train_eval_ratio=0.9,
            normal_weight_initialization=True)

        ffnn_mnist = FFNNMNIST(input_size, features)
        ffnn_mnist.do_train(root_path, hyper_parameters)