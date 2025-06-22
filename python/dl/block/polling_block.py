__author__ = "Patrick Nicolas"
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

from dl.block.neural_block import NeuralBlock
import torch.nn as nn
from typing import Self, AnyStr, List, Optional
from dl import ConvException


class PoolingBlock(NeuralBlock):
    def __init__(self, block_id: AnyStr, pooling_type: AnyStr, pooling_kernel: int, pooling_stride: int) -> None:
        self.pooling_type = pooling_type
        self.pooling_kernel = pooling_kernel
        self.pooling_stride = pooling_stride
        super(PoolingBlock, self).__init__(block_id, self.get_module())

    @classmethod
    def default(cls, block_id: AnyStr, pooling_type: AnyStr, pooling_kernel: int) -> Self:
        return cls(block_id, pooling_type, pooling_kernel, pooling_stride=1)

    def transpose(self, extra: Optional[nn.Module] = None) -> Self:
        raise ConvException('Cannot invert a pooling block')

    def get_modules(self) -> List[nn.Module]:
        match self.pooling_type:
            case 'MaxPool1d':
                return [nn.MaxPool1d(self.pooling_kernel, self.pooling_stride)]
            case 'MaxPool2d':
                return [nn.MaxPool2d(self.pooling_kernel, self.pooling_stride)]
            case 'AvgPool1d':
                return [nn.AvgPool1d(self.pooling_kernel, self.pooling_stride)]
            case 'AvgPool3d':
                return [nn.AvgPool2d(self.pooling_kernel, self.pooling_stride)]
            case _:
                raise ConvException(f'Pooling {self.pooling_type} is not supported')


