from unittest import TestCase
import math
from util.decorators import timeit
import logging
import python

@timeit
def procedure() -> float:
    _sum = 0.0
    for index in range(10000):
        _sum += math.exp(-index)
    return _sum


class TestDecorator(TestCase):

    def test_timeit(self):
        logging.info(procedure())

    def test_find_module_specs(self):
        from util import check_modules_availability
        modules = ['math', 'os', 'xyz']
        logging.info(check_modules_availability(modules))
