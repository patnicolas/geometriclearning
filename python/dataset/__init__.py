__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

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
