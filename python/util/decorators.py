__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import time
import constants


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        duration = time.time() - start
        constants.log_info(f'Duration: {duration}')
        return 0
    return wrapper
