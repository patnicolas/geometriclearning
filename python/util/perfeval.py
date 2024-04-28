__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import time
import constants

"""
    Compute the performance of the execution of a function
    @param func: Function to execute and timed
    @param args: Arguments for the function, func
"""


class PerfEval(object):
    def __init__(self, func, args: list = None):
        self.func = func
        self.args = args

    def eval(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        self.__time()
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            constants.log_info(f'Default tensor type: {torch.get_default_dtype()}')
            self.__time()
        else:
            constants.log_info(f'CUDA not available')

    def __time(self):
        start = time.time()
        if self.args is not None:
            self.func(self.args)
        else:
            self.func()
        duration = time.time() - start
        constants.log_info(f'Duration {duration} for {torch.get_default_dtype()}')