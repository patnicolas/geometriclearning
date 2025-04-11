__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch


def min_max_scaler(x: torch.Tensor) -> torch.Tensor:
    import traceback
    from sklearn.preprocessing import MinMaxScaler

    try:
        y = MinMaxScaler().fit_transform(x)
        z = torch.tensor(y, dtype=x.dtype)
        return z
    except Exception as e:
        traceback.print_exc()
        return x


class DatasetException(BaseException):
    def __init__(self, *args, **kwargs):  # real signature unknown
        super(DatasetException, self).__init__(args, kwargs)