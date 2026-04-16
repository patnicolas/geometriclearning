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

from typing import Tuple, List
from manim import *
import math

Rnge = Tuple[int, int] | Tuple[int, int, int]

colors = (BLUE, RED, YELLOW, WHITE)

def get_num_ticks(range: List[float]) -> int:
    return int((range[1]-range[0])/range[2])

def extract_num_digits(x: float) -> int:
    return len(str(abs(int(x))))

def next_multiple(x: float, n: int) -> int:
    return math.ceil(x/n)*n