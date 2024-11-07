import unittest

from dl.model.custom.ffnn_mnist import FfnnMnist
from dl.model.custom.vae_mnist import VAEMNIST
from dl.training.hyper_params import HyperParams
import torch.nn as nn


class VAEMNISTTest(unittest.TestCase):

    def test_train(self):
        input_size = 784
        features = [256, 128]
        root_path = '../../../../data/MNIST'

        hyper_parameters = HyperParams(
            lr=0.0004,
            momentum=0.97,
            epochs=860,
            optim_label='adam',
            batch_size=32,
            loss_function=nn.MSELoss(),
            drop_out=0.2,
            train_eval_ratio=0.9,
            normal_weight_initialization=False)
        try:
            ffnn_mnist = FfnnMnist(input_size, features)
            vae_mnist = VAEMNIST(ffnn_mnist, latent_size=12)
            print(repr(vae_mnist))
            vae_mnist.do_train(root_path, hyper_parameters)
        except Exception as e:
            print(str(e))


if __name__ == '__main__':
    unittest.main()