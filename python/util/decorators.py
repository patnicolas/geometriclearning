__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import time
import logging


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        duration = time.time() - start
        text = 'Duration' if len(args) == 0 else f'{args[0]}, duration'
        logging.info(f'{text=} {duration=}')
        return 0
    return wrapper
