_author__ = "Patrick Nicolas"
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

import pandas as pd
import pygwalker as pyg
from typing import AnyStr


class GWalker(object):
    def __init__(self, csv_file: AnyStr):
        df = pd.read_csv(csv_file)
        self.walker = pyg.walk(df)


if __name__ == '__main__':
    my_csv_file = '../input/items.csv'
    gwalker = GWalker(my_csv_file)
