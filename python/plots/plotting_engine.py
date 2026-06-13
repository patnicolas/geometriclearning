__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

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

from abc import ABC, abstractmethod
from typing import AnyStr, Self, Dict, Any
import numpy as np
import torch
from plots.plotting_config import PlottingConfig


class PlottingEngine(ABC):
    __slots__ = ['data_dict', 'plotting_config']

    def __init__(self, data_dict: Dict[AnyStr, np.array], plotting_config: PlottingConfig | AnyStr) -> None:
        self.data_dict = data_dict
        self.plotting_config = plotting_config if isinstance(plotting_config, PlottingConfig) \
            else PlottingConfig.build(plotting_config)

    @classmethod
    def build_single(cls, label: AnyStr, data: np.array, plotting_config: PlottingConfig | AnyStr) -> Self:
        return cls(data_dict={label: data}, plotting_config=plotting_config)

    @classmethod
    def build_from_torch(cls,
                         data_dict_torch: Dict[AnyStr, torch.Tensor],
                         plotting_config: PlottingConfig | AnyStr) -> Self:

        plotting_config = plotting_config if isinstance(plotting_config, PlottingConfig) \
            else PlottingConfig.build(plotting_config)
        data_dict = {k: v.numpy() for k,v in data_dict_torch.items()}
        return cls(data_dict=data_dict, plotting_config=plotting_config)

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def save(self, filename: AnyStr, ctx: Any) -> None:
        pass
