import time
from typing import AnyStr
import logging
collector = {}


def timeit(func):
    """ Decorator for timing execution of methods """
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        duration = '{:.3f}'.format(time.time() - start)
        key: AnyStr = f'{args[1]} {args[2]}'
        logging.info(f'{key=}\t{duration=} secs.')
        cur_list = collector.get(key)
        if cur_list is None:
            cur_list = [time.time() - start]
        else:
            cur_list.append(time.time() - start)
        collector[key] = cur_list
        return 0
    return wrapper
