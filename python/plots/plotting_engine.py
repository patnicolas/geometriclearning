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
from typing import AnyStr, Self
import numpy as np
import torch
from plots.plotting_config import PlottingConfig


class PlottingEngine(ABC):
    def __init__(self, data: np.array, plotting_config: PlottingConfig) -> None:
        self.data = data
        self.plotting_config = plotting_config

    @classmethod
    def build(cls, data: torch.Tensor, plotting_config_str: AnyStr) -> Self:
        plotting_config = PlottingConfig.build(plotting_config_str)
        return cls(data.numpy(), plotting_config)

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def save(self, filename: AnyStr) -> None:
        pass
