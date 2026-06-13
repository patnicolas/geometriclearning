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
