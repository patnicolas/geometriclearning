import unittest
from dl.block.conv.conv_output_size import ConvOutputSize, SeqConvOutputSize


class ConvOutputSizeTest(unittest.TestCase):
    def test_output_size_1(self):
        conv_output_size = ConvOutputSize(kernel_size=(4, 4),
                                          stride=(2, 2),
                                          padding=(1, 1),
                                          max_pooling_kernel=-1)
        output_size = conv_output_size(input_size=(28, 28))
        print(output_size)
        self.assertTrue(output_size[0] == 14)

    def test_output_size_2(self):
        conv_output_size = ConvOutputSize(kernel_size=(4, 4),
                                          stride=(2, 2),
                                          padding=(1, 1),
                                          max_pooling_kernel=2)
        output_size = conv_output_size(input_size=(28, 28))
        print(output_size)
        self.assertTrue(output_size[0] == 7)
        self.assertTrue(output_size[1] == 7)

    def test_seq_output_size_1(self):
        seq_conv_output_size = SeqConvOutputSize(
                            [ConvOutputSize(kernel_size=(4, 4),
                                            stride=(2, 2),
                                            padding=(1, 1),
                                            max_pooling_kernel=-1),
                             ConvOutputSize(kernel_size=(4, 4),
                                            stride=(2, 2),
                                            padding=(1, 1),
                                            max_pooling_kernel=-1)]
        )
        seq_output_size = seq_conv_output_size(input_size=(28, 28), out_channels=-1)
        print(seq_output_size)
        self.assertTrue(seq_output_size == 7*7)

    def test_seq_output_size_2(self):
        seq_conv_output_size = SeqConvOutputSize(
                            [ConvOutputSize(kernel_size=(4, 4),
                                            stride=(2, 2),
                                            padding=(1, 1),
                                            max_pooling_kernel=-1),
                             ConvOutputSize(kernel_size=(4, 4),
                                            stride=(2, 2),
                                            padding=(1, 1),
                                            max_pooling_kernel=-1)]
        )
        seq_output_size = seq_conv_output_size(input_size=(28, 28), out_channels=64)
        print(seq_output_size)
        self.assertTrue(seq_output_size == 64*7*7)

    def test_seq_output_size_3(self):
        seq_conv_output_size = SeqConvOutputSize(
                            [ConvOutputSize(kernel_size=(4, 4),
                                            stride=(2, 2),
                                            padding=(1, 1),
                                            max_pooling_kernel=2),
                             ConvOutputSize(kernel_size=(4, 4),
                                            stride=(2, 2),
                                            padding=(1, 1),
                                            max_pooling_kernel=2)]
        )
        seq_output_size = seq_conv_output_size(input_size=(28, 28), out_channels=64)
        print(seq_output_size)
        self.assertTrue(seq_output_size == 64*2*2)


