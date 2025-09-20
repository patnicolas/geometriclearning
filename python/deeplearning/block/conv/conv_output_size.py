__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard Library imports
from typing import Tuple, List
# Library imports
from deeplearning.block.conv import ConvDataType
__all__ = ['ConvOutputSize', 'SeqConvOutputSize']


class ConvOutputSize(object):
    """
    Class that wraps the computation of the size of the output of a convolutional neural block.
    math::
        W_{conv}[out] = \frac{W_{conv}[in]+2p-k}{s} +1
        H_{conv}[out]= \frac{H_{conv}[in]+2p-k}{s} +1

    Reference: https://patricknicolas.substack.com/p/reusable-neural-blocks-in-pytorch
    """
    def __init__(self,
                 kernel_size: ConvDataType,
                 stride: ConvDataType,
                 padding: ConvDataType,
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

    def __call__(self, input_size: ConvDataType) -> ConvDataType:
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

    def __layer_output_shapes(self, input_size: ConvDataType) -> ConvDataType:
        if isinstance(input_size, Tuple):
            return (self.__layer_output_shape(input_size=input_size[0], dim=0),
                    self.__layer_output_shape(input_size=input_size[1], dim=1))
        else:
            return self.__layer_output_shape(input_size=input_size, dim=0)

    def __pooling_output_shapes(self, input_size: ConvDataType) -> ConvDataType:
        if isinstance(input_size, Tuple):
            return (self.__pooling_output_shape(input_size=input_size[0]),
                    self.__pooling_output_shape(input_size=input_size[1]))
        else:
            return self.__pooling_output_shape(input_size=input_size)

    def __layer_output_shape(self, input_size: int, dim: int) -> int:
        assert 0 <= dim <= 1, f'Dimension {dim} for computing output channel is out of bounds (0, 1)'
        assert self.stride[dim] > 0, f'Stride {self.stride} should be > 0'

        stride = self.stride[dim]
        padding = self.padding[dim]
        kernel_size = self.kernel_size[dim]
        num = (input_size + 2 * padding - kernel_size)
        out_size = int(num / stride) + 1
        return out_size

    def __pooling_output_shape(self, input_size) -> int:
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



