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
import logging
# 3rd Party imports
import torch
import time
import python
__all__ = ['PerfEval']


class PerfEval(object):
    """
        Compute the performance of the execution of a function
    """
    def __init__(self, func, args: list = None):
        """
        Constructor for the Performance Evaluator

        @param func: Function to execute and timed
        @param args: Arguments for the function, func
        """
        self.func = func
        self.args = args

    def eval(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        self.__time()
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            logging.info(f'Default tensor type: {torch.get_default_dtype()}')
            self.__time()
        else:
            logging.info(f'CUDA not available')

    def __time(self):
        start = time.time()
        if self.args is not None:
            self.func(self.args)
        else:
            self.func()
        duration = time.time() - start
        logging.info(f'Duration {duration} for {torch.get_default_dtype()}')