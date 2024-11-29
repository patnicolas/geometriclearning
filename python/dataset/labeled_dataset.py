__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
from typing import AnyStr, Callable, Optional, List, Self
from python.dataset.dataset_exception import DatasetException
import numpy as np
import pandas as pd
from python.dataset.tdataset import TDataset
import logging
logger = logging.getLogger('dataset.LabeledDataset')

__all__ = ['LabeledDataset']


class LabeledDataset(TDataset):
    def __init__(self,
                 features: torch.Tensor,
                 labels: torch.Tensor,
                 transform: Optional[Callable] = None,
                 dtype: AnyStr = TDataset.default_float_type):
        """
        Constructor for a Tensor dataset
        @param features: Tensor containing features sets
        @type features: torch.Tensor
        @param labels: Torch tensor containing the labels
        @type labels: torch.Tensor
        @param transform: Apply pre-processing transform to input data
        @type transform: Callable transform
        @param dtype: Type of float used in computation
        @type dtype: str
        """
        # Make sure that the tensor are using the correct types
        features, labels = LabeledDataset.__update_dtype(features, labels, dtype)

        # Apply transform to the training and evaluation set if needed.
        self.features = transform(features) if transform else features
        self.labels = labels
        super(LabeledDataset, self).__init__(transform, dtype)

    @classmethod
    def from_numpy(cls,
                   features: np.array,
                   labels: np.array,
                   transform: Optional[Callable] = None,
                   dtype: AnyStr = TDataset.default_float_type) -> Self:
        """
        Create a Tensor features and labels dataset from a Numpy array
        @param features: Feature data as a Numpy array
        @type features: A Numpy array
        @param labels: Feature data as a Numpy array
        @type labels: A Numpy array
        @param transform: Pre-processing data transform
        @type transform: Optional Callable
        @param dtype: Type used for computation
        @type dtype: str
        @return: Instance of tensor dataset
        @rtype: LabeledDataset
        """
        return cls(torch.from_numpy(features), torch.from_numpy(labels), transform, dtype)

    @classmethod
    def from_list(cls,
                  features: List[List[float]],
                  labels: List[float],
                  transform: Optional[Callable] = None,
                  dtype: AnyStr = TDataset.default_float_type) -> Self:
        """
        Create a Tensor dataset from a Python list
        @param features: Feature values as list of floating point values
        @type features: List of Array of floats
        @param labels: Label values as list of floating point values
        @type labels: List of Array of floats
        @param transform: Pre-processing data transform
        @type transform:  Instance of Tensor
        @param dtype: Type used for computation
        @type dtype: str
        @return: Instance of tensor dataset
        @rtype: LabeledDataset
        """
        assert len(features) > 0, 'Cannot create a features tensor dataset from undefined Python list'
        assert len(labels) > 0, 'Cannot create a label tensor dataset from undefined Python list'
        return cls(torch.tensor(features), torch.Tensor(labels), transform, dtype)

    @classmethod
    def from_df(cls,
                df_train: pd.DataFrame,
                df_eval: pd.DataFrame,
                transform: Optional[Callable] = None,
                dtype: AnyStr = TDataset.default_float_type) -> Self:
        """
        Create a Tensor dataset from a Pandas data frame
        @param df_train: Pandas data frame for training data
        @type df_train: pd.DataFrame
        @param df_eval: Pandas data frame for evaluation or test data
        @type df_eval: pd.DataFrame
        @param transform: Optional pre-processing transform
        @type transform: Optional[Callable]
        @param dtype: Type used for computation
        @type dtype: str
        @return: Instance of labeled dataset
        @rtype: LabeledDataset
        """
        np_train: np.array = df_train.to_numpy()
        np_eval: np.array = df_eval.to_numpy()
        return cls(torch.from_numpy(np_train), torch.from_numpy(np_eval), transform, dtype)

    @classmethod
    def from_file(cls,
                  filename: AnyStr,
                  features: List[AnyStr],
                  label: AnyStr,
                  transform: Optional[Callable] = None,
                  dtype: AnyStr = 'float64') -> Self:
        """
        Generate a Tensor data set from a JSON or CSV file..
        @param filename: Name of the file (relative or absolute)
        @type filename: Str
        @param features: List of column names used in the training data set
        @type features: List[str]
        @param label: List of column names used in the label data set
        @type label: List[str]
        @param transform: Callable pre-processing data transformation on the column values
        @type transform: Callable
        @param dtype: Type used for computation
        @type dtype: str
        @return: Labeled tensor dataset
        @rtype: LabeledDataset
        """
        try:
            df = LabeledDataset.data_frame(filename)
            TDataset._display(df, 20, features)
            sub_set = set(features)
            if sub_set.intersection(set(df.columns)) != sub_set:
                raise DatasetException(f'One of the column {str(features)} is not supported!')

            return LabeledDataset.from_df(df[features], df[label], transform, dtype)
        except Exception as e:
            logger.error(f'Unknown error {str(e)}')
            raise DatasetException(f'Unknown error {str(e)}')

    def __len__(self) -> int:
        """
        Override the computation the size of the data
        @return: Number of data points in the data set
        @rtype: int
        """
        return len(self.features)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """
        Override the selection of item in the data set. Throws an IndexError if index is out of bounds
        @param idx: Index in the data point selected
        @type idx: int
        @return: Item as a Torch tensor
        @rtype: torch.Tensor
        """
        if idx >= len(self.features):
            raise IndexError(f'getitem index {idx} should be < {len(self.features)}')
        return self.features[idx], self.labels[idx]

    def __repr__(self):
        return f'Features:\n{str(self.features.numpy())}\nLabels:\n{str(self.labels.numpy())}' \
               f'\nTransform:{self.transform}\nDtype: {self.dtype}'

    """ -------------------------  Private Helper Methods ------------------------ """
    @staticmethod
    def __update_dtype(
            features: torch.Tensor,
            labels: torch.Tensor,
            dtype: AnyStr) -> (torch.Tensor, torch.Tensor):
        default_type = TDataset._torch_type(dtype)
        if features.dtype != default_type:
            features = features.to(default_type)
        if labels.dtype != default_type:
            labels = labels.to(default_type)
        return features, labels
