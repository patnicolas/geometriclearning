__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, AnyStr, List
import pandas as pd
from python.dataset.datasetexception import DatasetException
import logging
logger = logging.getLogger('dataset.TDataset')


class TDataset(Dataset):
    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform

    @staticmethod
    def data_frame(filename: AnyStr) -> pd.DataFrame:
        try:
            ext = TDataset._extract_extension(filename)
            match ext:
                case '.csv':
                    df = pd.read_csv(filename)
                case '.json':
                    df = pd.read_json(filename)
                case _:
                    raise DatasetException(f'Extension {ext} is not supported!')
            return df

        except FileNotFoundError as e:
            logger.error(f'Filename {filename} not found')
            raise DatasetException(f'Filename {filename} not found {str(e)}')
        except Exception as e:
            logger.error(f'Unknown error {str(e)}')
            raise DatasetException(f'Unknown error {str(e)}')

    """  --------------------  Private Helper Methods -------------------------- """

    @staticmethod
    def _extract_extension(filename: AnyStr) -> AnyStr:
        import os
        _, file_ext = os.path.splitext(filename)
        return file_ext

    @staticmethod
    def _display(df: pd.DataFrame, num: int, column_names: Optional[List[AnyStr]] = None):
        from IPython.display import display
        show_df = df[column_names].head(num) if column_names is not None else df.head(num)
        display(show_df)