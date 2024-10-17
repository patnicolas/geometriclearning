
import unittest
from dl.model.custom.conv_mnist import ConvMNIST
from dl.block import ConvException
from dl.dlexception import DLException


class ConvMNISTTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init(self):
        input_size = 28
        in_channels = [1, 32]
        kernel_size = [3, 3]
        padding_size = [0, 0]
        stride_size = [1, 1]
        max_pooling_kernel = 2
        out_channels = 64

        try :
            conv_MNIST_instance = ConvMNIST(
                input_size,
                in_channels,
                kernel_size,
                padding_size,
                stride_size,
                max_pooling_kernel,
                out_channels)
            print(repr(conv_MNIST_instance))
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)

    def test_train(self):
        input_size = 28
        in_channels = [1, 32]
        kernel_size = [3, 3]
        padding_size = [0, 0]
        stride_size = [1, 1]
        max_pooling_kernel = 2
        out_channels = 64
        root_path = '../../../../data/MNIST'

        try:
            conv_MNIST_instance = ConvMNIST(
                input_size,
                in_channels,
                kernel_size,
                padding_size,
                stride_size,
                max_pooling_kernel,
                out_channels)
            print(repr(conv_MNIST_instance))
            conv_MNIST_instance.train_(root_path)
            self.assertTrue(True)
        except ConvException as e:
            print(str(e))
            self.assertTrue(False)
        except DLException as e:
            print(str(e))
            self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()

