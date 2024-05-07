__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Callable, AnyStr, List
import pandas as pd
import numpy as np
from python.dataset.datasetexception import DatasetException
import traceback
import logging
logger = logging.getLogger('dataset.TDataset')


def min_max_scaler(x: torch.Tensor) -> torch.Tensor:
    try:
        y = MinMaxScaler().fit_transform(x)
        z = torch.tensor(y, dtype=x.dtype)
        return z
    except:
        traceback.print_exc()
        return x


class TDataset(Dataset):
    supported_dtypes = ['float64', 'float32', 'float16', 'double']
    default_float_type = 'float32'

    def __init__(self, transform: Optional[Callable] = None, dtype: AnyStr = default_float_type):
        self.transform = transform
        # Make sure the type is supported
        if dtype not in TDataset.supported_dtypes:
            raise DatasetException(f'Data type {dtype} is not supported')
        self.dtype = dtype

    @staticmethod
    def _torch_type(dtype: AnyStr) -> torch:
        match dtype:
            case 'float64': return torch.float64
            case 'float32': return torch.float32
            case 'float16': return torch.float16
            case 'double': return torch.double
            case 'long': return torch.long
            case _: return torch.float32

    @staticmethod
    def numpy_type(dtype: AnyStr) -> np:
        match dtype:
            case 'float64': return np.float64
            case 'float32': return np.float32
            case 'float16': return np.float16
            case 'double': return np.float64
            case _: return np.float32

    @staticmethod
    def data_frame(filename: AnyStr) -> pd.DataFrame:
        try:
            ext = TDataset._extract_extension(filename)
            match ext:
                case '.csv':
                    df = pd.read_csv(filename, dtype=float)
                case '.json':
                    df = pd.read_json(filename, dtype=float)
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