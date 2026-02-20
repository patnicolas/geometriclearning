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


# Standard Library imports
from typing import AnyStr
# 3rd Party imports
import pandas as pd
import json
import pickle
__all__ = ['IOUtil']


class IOUtil(object):
    """
        Generic wrapper for logging data
    """
    def __init__(self, path: AnyStr) -> None:
        """
        Constructor for a generic utility for I/O operations
        @param path: Path for the local logging file
        """
        self.path = path

    def to_lines(self) -> list:
        with open(self.path, 'r') as f:
            lines = f.readlines()
        return lines

    def from_text(self, content: str):
        with open(self.path, 'w') as f:
            f.write(content)

    def to_json(self) -> dict:
        lines = self.to_lines()
        content = '\n'.join(lines)
        return json.loads(content)

    def to_pickle(self, lst: list):
        with open(self.path, "wb") as f:
            pickle.dump(lst, f)

    def from_pickle(self) -> list:
        with open(self.path, "rb") as f:
            return pickle.load(f)

    def to_text(self) -> str:
        return ''.join(self.to_lines())

    def to_dataframe(self) -> pd.DataFrame:
        pd.read_json(self.path)

    @staticmethod
    def lines_to_json(lines: list) -> dict:
        content = '\n'.join(lines)
        return json.loads(content)

    @staticmethod
    def model_id_s3path(model_id: str) -> str:
        return model_id.replace('-', '/')

    @staticmethod
    def s3path_model_id(s3_path: str) -> str:
        return s3_path.replace('/', '-')


