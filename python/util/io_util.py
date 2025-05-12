__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import pandas as pd
import json
import pickle

"""
    Generic wrapper for logging data 
    @param path: Path for the local logging file
"""


class IOUtil(object):
    def __init__(self, path):
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


