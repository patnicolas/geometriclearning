__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import AnyStr, List, Optional, Tuple

from dl.model.conv_model import ConvModel
from dl.block.conv import Conv3DataType
from dl.block.conv.conv_3d_block import Conv3dBlock
from dl.block.mlp_block import MLPBlock



class Conv2dModel(ConvModel):
    def __init__(self,
                 model_id: AnyStr,
                 input_size: Conv3DataType,
                 conv_blocks: List[Conv3dBlock],
                 mlp_blocks: Optional[List[MLPBlock]] = None) -> None:
        super(Conv2dModel, self).__init__(model_id, input_size, conv_blocks, mlp_blocks)


