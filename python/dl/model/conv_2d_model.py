__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import AnyStr, List, Optional, Tuple, Dict, Any, Self

from dl.model.conv_model import ConvModel
from dl.model.mlp_model import MLPBuilder
from dl.model.neural_model import NeuralBuilder
from dl.block.conv import Conv2DataType
from dl.block.conv.conv_2d_block import Conv2dBlock
from dl.block.mlp_block import MLPBlock


class Conv2dModel(ConvModel):
    def __init__(self,
                 model_id: AnyStr,
                 input_size: Conv2DataType,
                 conv_blocks: List[Conv2dBlock],
                 mlp_blocks: Optional[List[MLPBlock]] = None) -> None:
        """
        Default constructor for Convolutional Model
        @param model_id: Identifier for model
        @type model_id: AnyStr
        @param input_size: Size of input (i.e. 64, 28x28, ...)
        @type input_size: int or Tuple[Int, Int]
        @param conv_blocks: List of convolutional block
        @type conv_blocks: List[Conv2dBlock]
        @param mlp_blocks: Optional list of MLP blocks
        @type mlp_blocks: List[MLPBlock]
        """
        super(Conv2dModel, self).__init__(model_id, input_size, conv_blocks, mlp_blocks)


class Conv2dBuilder(NeuralBuilder):
    """
    Builder for the 2-dimensional convolutional neural network.
    The convolutional neural model is built from a dictionary of configuration parameters
    for which  the keys are predefined. The model is iteratively created by call to method set
    defined in the base class NeuralBuilder

    The constructor define defaults value for activation (nn.ReLU()), stride, padding,
    enabling batch normalization and drop_out (no dropout).

    Reference: https://patricknicolas.substack.com/p/modular-deep-learning-models-with
    """
    def __init__(self, model_attributes: Dict[AnyStr, Any]) -> None:
        """
        Constructor for the builder of a 2-dimensional neural network using default
        set of keys (name of configuration parameters) and default value for activation
        module, stride, padding, enabling batch normalization and no dropout
        @param model_attributes: Dictionary of model attributes
        @type model_attributes: Dict[AnyStr, Any]
        """
        super(Conv2dBuilder, self).__init__(model_attributes)

    def build(self) -> Conv2dModel:
        """
        Build the 2-dimensional convolutional network from a dictionary of configuration
        parameters in three steps:
        1- Generate the convolutional neural block from the configuration parameters
        2- Generate the MLP neural blocks from the configuration
        3- Validate the model
        @return: 2-dimensional convolutional model instance
        @rtype: Conv2dModel
        """
        # Instantiate the model from the dictionary of Configuration parameters
        model_id = self.model_attributes['model_id']
        # Generate the convolutional neural blocks from the configuration attributes dictionary
        conv_blocks = self.__create_conv_blocks()
        # Generate the fully connected blocks from the configuration attributes dictionary
        mlp_blocks = self.__create_mlp_blocks()
        # Validation
        Conv2dBuilder.__validate(conv_blocks, mlp_blocks, self.model_attributes['input_size'])
        return Conv2dModel(model_id, self.model_attributes['input_size'], conv_blocks, mlp_blocks)

    def __create_conv_blocks(self) -> List[Conv2dBlock]:
        conv_2d_blocks = []
        in_channel = self.model_attributes['in_channels_list'][0]
        for idx in range(1, len(self.model_attributes['in_channels_list'])):
            conv_block = Conv2dBlock.build_from_params(
                block_id=f'{self.model_attributes["model_id"]}-{idx}',
                in_channels=in_channel,
                out_channels=self.model_attributes['in_channels_list'][idx],
                kernel_size=self.model_attributes['kernel_size'],
                stride=self.model_attributes['stride'],
                padding=self.model_attributes['padding'],
                batch_norm=self.model_attributes['is_batch_norm'],
                max_pooling_kernel=self.model_attributes['max_pool_kernel'],
                activation=self.model_attributes['activation'],
                bias=self.model_attributes['bias'],
                drop_out=self.model_attributes['drop_out'])
            conv_2d_blocks.append(conv_block)
            in_channel = self.model_attributes['in_channels_list'][idx]
        return conv_2d_blocks

    def __create_mlp_blocks(self) -> List[MLPBlock]:
        mlp_builder = MLPBuilder(self.model_attributes)
        mlp_builder.set(key='in_features_list', value=self.model_attributes['in_features_list'])
        mlp_builder.set(key='activation', value=self.model_attributes['activation'])
        mlp_builder.set(key='drop_out', value=self.model_attributes['drop_out'])
        mlp_builder.set(key='output_activation', value=self.model_attributes['output_activation'])
        mlp_model = mlp_builder.build()
        return mlp_model.neural_blocks

    @staticmethod
    def __validate(conv_blocks: List[Conv2dBlock],
                   mlp_blocks: List[MLPBlock],
                   input_size: Conv2DataType) -> None:
        """
        Test if the layout/configuration of convolutional neural blocks and feed-forward neural blocks
        are valid
        @param conv_blocks: List of Convolutional blocks which layout is to be evaluated
        @type conv_blocks: List[ConvBlock]
        @param mlp_blocks:  List of neural blocks which layout is to be evaluated
        @type mlp_blocks: List[MLPBlock]
        @param input_size: Input size as int (1D) or Tuple (2D)
        """
        from dl.model.mlp_model import MLPBuilder
        assert conv_blocks, 'This convolutional model has not defined neural blocks'
        Conv2dBuilder.validate_conv(conv_blocks, input_size)
        MLPBuilder.validate(mlp_blocks)

    """ ----------------------------   Private helper methods --------------------------- """

    @staticmethod
    def validate_conv(conv_blocks: List[Conv2dBlock], input_size: Conv2DataType) -> None:
        assert len(conv_blocks) > 0, \
            "Convolutional network needs at least one layer"

        for index in range(len(conv_blocks) - 1):
            # 1. Validate the in-channel and out-channels
            next_in_channels = conv_blocks[index + 1].get_in_channels()
            this_out_channels = conv_blocks[index].get_out_channels()
            assert next_in_channels == this_out_channels, \
                f'Layer {index} input_tensor != layer {index + 1} output'

            # 2. Validate the in-channel and out-channels
            this_output_shape_1 = Conv2dBuilder.__get_out_shape_dim(
                conv_block=conv_blocks[index],
                input_size=input_size,
                dim=0
            )
            this_output_shape_2 = Conv2dBuilder.__get_out_shape_dim(
                conv_block=conv_blocks[index],
                input_size=input_size,
                dim=1
            )
            next_input_shape = input_size
            assert (this_output_shape_1 == next_input_shape[0] and
                    this_output_shape_2 == next_input_shape[1],
                    (f'This output shape {str(this_output_shape_1)} should equal '
                     f'next input shape {str(next_input_shape[0])}'))

    @staticmethod
    def __get_out_shape_dim(conv_block: Conv2dBlock, input_size: Conv2DataType, dim: int) -> int:
        stride = conv_block.modules_list[0].stride[dim]
        padding = conv_block.modules_list[0].padding[dim]
        kernel_size = conv_block.modules_list[0].kernel_size[dim]
        num = (input_size[dim] + 2 * padding - kernel_size)
        return int(num / stride) + 1

    @staticmethod
    def output_size(input_size: Tuple[int, int],
                    kernel_size: Tuple[int, int],
                    padding: Tuple[int, int],
                    stride: Tuple[int, int]) -> Tuple[int, int]:
        num_w = (input_size[0] + 2 * padding[0] - kernel_size[0])
        num_h = (input_size[1] + 2 * padding[1] - kernel_size[1])
        return int(num_w / stride[0]) + 1, int(num_h / stride[1]) + 1

