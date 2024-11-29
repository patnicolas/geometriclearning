__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
from typing import Callable, Optional, Self, List, AnyStr
import numpy as np
import pandas as pd
from python.dataset.dataset_exception import DatasetException
from python.dataset.tdataset import TDataset
import logging
logger = logging.getLogger('dataset.UnlabeledDataset')


"""
Generic dataset loaded from a torch tensor, a Numpy array, a Python array or a Panda Dataframe.
All exceptions thrown by the various methods are consolidated into DatasetException
"""


class UnlabeledDataset(TDataset):
    def __init__(self,
                 data: torch.Tensor,
                 transform: Optional[Callable] = None,
                 dtype: AnyStr = TDataset.default_float_type):
        """
        Constructor for a Tensor dataset
        @param data: Tensor containing data
        @type data: Torch tensor
        @param transform: Apply pre-processing transform to input data
        @type transform: Callable transform
        @param dtype: Type of float used in computation
        @type dtype: str
        """
        # Make sure that the tensor are using the correct types
        data = UnlabeledDataset.__update_dtype(data, dtype)
        self.data = transform(data) if transform else data
        super(UnlabeledDataset, self).__init__(transform, dtype)

    @classmethod
    def from_numpy(cls,
                   data: np.array,
                   transform: Optional[Callable] = None,
                   dtype: AnyStr = TDataset.default_float_type) -> Self:
        """
        Create a Tensor dataset from a Numpy array
        @param data: Data as a Numpy array
        @type data: A Numpy array
        @param transform: Pre-processing data transform
        @type transform: Optional Callable
        @param dtype: Type of float used in computation
        @type dtype: str
        @return: Instance of tensor dataset
        @rtype: TensorDataset
        """
        return cls(torch.from_numpy(data), transform, dtype)

    @classmethod
    def from_list(cls,
                  data: List[List[float]],
                  transform: Optional[Callable] = None,
                  dtype: AnyStr = TDataset.default_float_type) -> Self:
        """
        Create a Tensor dataset from a Python list
        @param data: Data as list of floating point values
        @type data: List of Array of floats
        @param transform: Pre-processing data transform
        @type transform: Optional Callable
        @param dtype: Type of float used in computation
        @type dtype: str
        @return: Instance of unlabeled tensor dataset
        @rtype:  UnlabeledDataset
        """
        assert len(data) > 0, 'Cannot create a tensor dataset from undefined Python list'
        return cls(torch.Tensor(data), transform, dtype)

    @classmethod
    def from_df(cls,
                df: pd.DataFrame,
                transform: Optional[Callable] = None,
                dtype: AnyStr = TDataset.default_float_type) -> Self:
        """
        Create a Tensor dataset from a Pandas data frame
        @param df: Pandas data frame
        @type df: pd.DataFrame
        @param transform: Pre-processing data transform
        @type transform: Optional Callable
        @param dtype: Type of float used in computation
        @type dtype: str
        @return: Instance of unlabeled dataset
        @rtype: UnlabeledDataset
        """
        data: np.array = df.to_numpy()
        return cls(torch.from_numpy(data), transform, dtype)

    @classmethod
    def from_file(cls,
                  filename: AnyStr,
                  columns: List[AnyStr],
                  transform: Optional[Callable] = None,
                  dtype: AnyStr = TDataset.default_float_type) -> Self:
        """
        Generate a Tensor data set from a JSON or CSV file..
        @param filename: Name of the file (relative or absolute)
        @type filename: Str
        @param columns: List of column names used in the data set
        @type columns: List[str]
        @param transform: Pre-processing data transform
        @type transform: Optional Callable
        @param dtype: Type of float used in computation
        @type dtype: str
        @return: Unlabeled tensor dataset
        @rtype: UnlabeledDataset
        """
        try:
            df = UnlabeledDataset.data_frame(filename)
            TDataset._display(df, 20, columns)
            sub_set = set(columns)
            if sub_set.intersection(set(df.columns)) != sub_set:
                raise DatasetException(f'One of the column {str(columns)} is not supported!')
            return UnlabeledDataset.from_df(df, transform, dtype)
        except Exception as e:
            logger.error(f'Unknown error {str(e)}')
            raise DatasetException(f'Unknown error {str(e)}')

    def __len__(self) -> int:
        """
        Override the computation the size of the data
        @return: Number of data points in the data set
        @rtype: int
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Override the selection of item in the data set. Throws an IndexError if index is out of bounds
        @param idx: Index in the data point selected
        @type idx: int
        @return: Item as a Torch tensor
        @rtype: torch.Tensor
        """
        if idx >= len(self.data):
            raise IndexError(f'getitem index {idx} should be < {len(self.data)}')
        # debug
        data_pt = self.data[idx]
        return self.transform(data_pt) if self.transform else data_pt

    def __repr__(self):
        return f'Data:\n{str(self.data.numpy())}\nTransform:{self.transform}'

    """ --------------------  Private helper methods ---------------------- """

    @staticmethod
    def __update_dtype(data: torch.Tensor, dtype: AnyStr) -> torch.Tensor:
        default_type = TDataset._torch_type(dtype)
        if data.dtype != default_type:
            data = data.to(default_type)
        return data

