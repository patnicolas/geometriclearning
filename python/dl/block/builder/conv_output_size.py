__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."
__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import Tuple, List


class ConvOutputSize(object):
    def __init__(self,
                 kernel_size: int | Tuple[int, int],
                 stride: int | Tuple[int, int],
                 padding: int | Tuple[int, int],
                 max_pooling_kernel: int) -> None:
        """
        Constructor for the computation of the shape of the output for convolutional layer (kernel, stride, padding)
        and pooling layer (pooling kernel and stride) given an input shape
        Note: It is assumed that the stride for the pooling layer is identical as the side of its kernel

        @param kernel_size: Size of the kernel for the convolutional layer
        @type kernel_size: int
        @param stride: Stride for the convolutional layer
        @type stride: int
        @param padding: Padding for the convolutional layer
        @type padding: int
        @param max_pooling_kernel: Size of the kernel for the pooling layer
        @type max_pooling_kernel: int
        """
        super(ConvOutputSize, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_pooling_kernel = max_pooling_kernel

    def __call__(self, input_size: int | Tuple[int, int]) -> int | Tuple[int, int]:
        """
        Generic method to compute the output shape through a pair of convolutional and pooling layers
        given an input shape. The input shape can be an int for 1 dimension convolution or a tuple for
        a 2 dimension convolution
        @param input_size: Input shape (height or width)
        @type input_size: either int or Tuple
        @return: Output shape
        @rtype: either int or Tuple
        """
        next_input_size = self.__layer_output_shapes(input_size)
        return self.__pooling_output_shapes(next_input_size) if self.max_pooling_kernel > 0 else next_input_size

    """ -----------------------------    Private helper methods ------------------------ """

    def __layer_output_shapes(self, input_size: int | Tuple[int, int]) -> int | Tuple[int, int]:
        if isinstance(input_size, Tuple):
            return (self.__layer_output_shape(input_size=input_size[0], dim=0),
                    self.__layer_output_shape(input_size=input_size[1], dim=1))
        else:
            return self.__layer_output_shape(input_size=input_size, dim=0)

    def __pooling_output_shapes(self, input_size: int | Tuple[int, int]) -> int | Tuple[int, int]:
        if isinstance(input_size, Tuple):
            return (self.__pooling_output_shape(input_size=input_size[0], dim=0),
                    self.__pooling_output_shape(input_size=input_size[1], dim=1))
        else:
            return self.__pooling_output_shape(input_size=[input_size], dim=0)

    def __layer_output_shape(self, input_size: int, dim: int) -> int:
        assert 0 <= dim <= 1, f'Dimension {dim} for computing output channel is out of bounds (0, 1)'
        assert self.stride[dim] > 0, f'Stride {self.stride} should be > 0'

        stride = self.stride[dim]
        padding = self.padding[dim]
        kernel_size = self.kernel_size[dim]
        num = (input_size + 2 * padding - kernel_size)
        out_size = int(num / stride) + 1
        return out_size

    def __pooling_output_shape(self, input_size: int, dim: int) -> int:
        out_size = int((input_size - self.max_pooling_kernel) + 1)
        return out_size


class SeqConvOutputSize(object):
    def __init__(self, conv_output_sizes: List[ConvOutputSize]) -> None:
        """
        Constructor for the computation of the shape of an input element through a sequence of pair
        convolutional layer and pooling layer.

        @param conv_output_sizes: List of transformation of
        @type conv_output_sizes: List
        """
        self.conv_output_sizes = conv_output_sizes

    def __call__(self, input_size: int | Tuple[int, int], out_channels: int = -1) -> int:
        """
        Generic method to compute the output shape through a sequence of pairs of convolutional and pooling layers
        given an input size. The input size can be an int for 1 dimension convolution or a tuple for
        a 2 dimension convolution.
        The out_channels parameters is used to flatten the output if it is > 0. By default, the output is not flattened

        @param input_size: Input size (height or width)
        @type input_size: either int or Tuple
        @param out_channels: Output channels used to flatten the convolution if it is defined > 0
        @type out_channels: int
        @return: Output size
        @rtype: either int or Tuple
        """
        # execute the sequence of transformation for the input size
        next_input_size = input_size
        for conv_output_size in self.conv_output_sizes:
            next_input_size = conv_output_size(next_input_size)
        return next_input_size



