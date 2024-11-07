import unittest
from dl.model.custom.ffnn_mnist import FfnnMnist
from dl.training.hyper_params import HyperParams
import torch.nn as nn


class FFNNMISTTest(unittest.TestCase):

    def test_init(self):
        features = [256, 128]
        input_size = 784
        ffnn_mnist = FfnnMnist(input_size, features)
        print(repr(ffnn_mnist))
        self.assertTrue(True)

    def test_train(self):
        input_size = 784
        features = [256, 128, 64]
        root_path = '../../../../data/MNIST'
        metric_label = 'FFNN - MNIST'
        hyper_parameters = HyperParams(
            lr=0.0005,
            momentum=0.95,
            epochs=24,
            optim_label='adam',
            batch_size=16,
            loss_function=nn.CrossEntropyLoss(),
            drop_out=0.20,
            train_eval_ratio=0.9,
            normal_weight_initialization=False)

        ffnn_mnist = FfnnMnist(input_size, features)
        ffnn_mnist.do_train(root_path, hyper_parameters, metric_label)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
