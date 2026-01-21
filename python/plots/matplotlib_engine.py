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

from typing import AnyStr
import numpy as np
from plots.plotting_engine import PlottingEngine
from plots.plotting_config import PlottingConfig


class MatplotlibEngine(PlottingEngine):
    def __init__(self, data: np.array, plotting_config: PlottingConfig, fig) -> None:
        super(MatplotlibEngine, self).__init__(data, plotting_config)
        self.fig = fig

    def render(self) -> None:
        pass

    def save(self, filename: AnyStr) -> None:
        if self.plotting_config.filename is not None:
            self.fig.savefig(f'{self.plotting_config.filename}.png')
