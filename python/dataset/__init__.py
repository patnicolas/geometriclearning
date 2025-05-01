__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch


def min_max_scaler(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper method for applying min-max scaler as
    x -> (x - min)/(max - min)
    The method invoke the scikit-learn class MinMaxScaler. Any exception is converted to a
    dataset exception as it is related to the distribution of the input data.
    @param x: Input tensor
    @type x: torch.Tensor
    @return: Tensor normalized using min-max scaler
    @rtype: torch.Tensor
    """
    import traceback
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.exceptions import NotFittedError

    try:
        y = MinMaxScaler().fit_transform(x)
        z = torch.tensor(y, dtype=x.dtype)
        return z
    except ValueError | NotFittedError | TypeError as e:
        traceback.print_exc()
        raise DatasetException(f'MinMaxScaler failed {e}')


class DatasetException(BaseException):
    """
    Exception related to data sets.
    """
    def __init__(self, *args, **kwargs) -> None:  # real signature unknown
        super(DatasetException, self).__init__(args, kwargs)
