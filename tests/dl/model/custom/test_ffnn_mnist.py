import unittest
from dl.model.custom.ffnn_mnist import FFNNMNIST

class FFNNMISTTest(unittest.TestCase):

    def test_init(self):
        in_channels = [256, 128]
        input_size = 28
        ffnn_mnist = FFNNMNIST(input_size, in_channels)
        print(repr(ffnn_mnist))