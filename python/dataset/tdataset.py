__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Callable, AnyStr, List, Tuple
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


"""
Generic type for any labeled or unlabeled data set 
"""


class TDataset(Dataset):
    supported_dtypes = ['int16', 'int32', 'int64', 'float64', 'float32', 'float16', 'double']
    default_float_type = 'float32'

    def __init__(self, transform: Optional[Callable] = None, dtype: AnyStr = default_float_type):
        """
        Generic constructor for any type (labeled, unlabeled) data set
        @param transform: Transformation to be optionally applied to input data
        @type transform: Optional Callable
        @param dtype: Type of data (float, int,...)
        @type dtype: Str
        """
        # Make sure the type is supported
        if dtype not in TDataset.supported_dtypes:
            raise DatasetException(f'Data type {dtype} is not supported')
        self.transform = transform
        self.dtype = dtype

    @staticmethod
    def numpy_type(dtype: AnyStr) -> np:
        """
        Extract the numpy type from a string type
        @param dtype: String representation of type
        @type dtype: Str
        @return: Numpy data type
        @rtype: np
        """
        match dtype:
            case 'float64': return np.float64
            case 'float32': return np.float32
            case 'float16': return np.float16
            case 'double': return np.float64
            case _: return np.float32

    @staticmethod
    def data_frame(filename: AnyStr) -> pd.DataFrame:
        """
        Load a Pandas data frame from file. An exception is thrown if the file is not find,
        the extension is not supported or the Torch tensor extracted from the pt file is not
        a tensor
        @param filename: Name of the file
        @type filename: str
        @return: Pandas data frame
        @rtype: DataFrame
        """
        try:
            ext = TDataset._extract_extension(filename)
            match ext:
                case '.csv':
                    df = pd.read_csv(filename, dtype=float)
                case '.json':
                    df = pd.read_json(filename, dtype=float)
                case '.pt':
                    data = torch.load(filename)
                    if type(data) != torch.Tensor:
                        raise DatasetException(f'Incorrect data type for {filename}!')
                    df = pd.DataFrame(data.numpy())
                case _:
                    raise DatasetException(f'Extension {ext} is not supported!')
            return df

        except FileNotFoundError as e:
            logger.error(f'Filename {filename} not found')
            raise DatasetException(f'Filename {filename} not found {str(e)}')
        except Exception as e:
            logger.error(f'Unknown error {str(e)}')
            raise DatasetException(f'Unknown error {str(e)}')

    @staticmethod
    def torch_to_df(x: torch.Tensor) -> (pd.DataFrame, Tuple[int]):
        """
        Convert a torch tensor into a Pandas data frame. The tensor input is reshaped if
        the dimension > 2. The first dimension is preserved while all other dimension are squeezed.
        @param x: Input Torch tensor
        @type x: torch.Tensor
        @return: Tuple (Pandas data frame, Original shape of the tensor)
        @rtype: Tuple[pdDataFrame, Tuple[int])
        """
        if len(x.shape) == 0:
            raise DatasetException(f'Shape of tensor {x.shape} is not supported')
        elif len(x.shape) > 2:
            y = x.reshape(x.shape[0], -1)
        else:
            y = x
        return pd.DataFrame(y.numpy()), x.shape

    @staticmethod
    def torch_to_dfs(filename: AnyStr):
        """
        Convert a sequence of torch tensor contained in a foiled a list of pair (Pandas data frame,
        original shape of the tensor).
        An exception is thrown if the file is not find or the file is not a pt format or the type is not
        a tensor, a tuple of tensors or a list of tensors
        @param filename: Name of the file containing the tensors
        @type filename: str
        @return: List of pair (Data Frame, original shape of tensor)
        @rtype: List
        """
        try:
            ext = TDataset._extract_extension(filename)
            assert ext == '.pt'
            data = torch.load(filename)
            if type(data) == torch.Tensor:
                dfs = [TDataset.torch_to_df(data)]
            elif type(data) == tuple:
                dfs = [TDataset.torch_to_df(x) for x in data]
            elif type(data) == list:
                dfs = [TDataset.torch_to_df(x) for x in data]
            else:
                raise DatasetException(f'Incorrect data type for {filename}!')
            return dfs
        except FileNotFoundError as e:
            logger.error(f'Filename {filename} not found')
            raise DatasetException(f'Filename {filename} not found {str(e)}')
        except Exception as e:
            logger.error(f'Unknown error {str(e)}')
            raise DatasetException(f'Unknown error {str(e)}')

    """  --------------------  Protected/Private Helper Methods -------------------------- """

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
    def _extract_extension(filename: AnyStr) -> AnyStr:
        import os
        _, file_ext = os.path.splitext(filename)
        return file_ext

    @staticmethod
    def _display(df: pd.DataFrame, num: int, column_names: Optional[List[AnyStr]] = None):
        from IPython.display import display
        show_df = df[column_names].head(num) if column_names is not None else df.head(num)
        display(show_df)