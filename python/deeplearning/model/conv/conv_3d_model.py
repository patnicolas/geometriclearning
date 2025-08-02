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

from typing import AnyStr, List, Optional
from deeplearning.model.conv.conv_model import ConvModel
from deeplearning.block.conv import Conv3DataType
from deeplearning.block.conv.conv_3d_block import Conv3dBlock
from deeplearning.block.mlp.mlp_block import MLPBlock
__all__ = ['Conv3dModel']



class Conv3dModel(ConvModel):
    def __init__(self,
                 model_id: AnyStr,
                 input_size: Conv3DataType,
                 conv_blocks: List[Conv3dBlock],
                 mlp_blocks: Optional[List[MLPBlock]] = None) -> None:
        super(Conv3dModel, self).__init__(model_id, input_size, conv_blocks, mlp_blocks)


