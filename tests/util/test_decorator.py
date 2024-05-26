from unittest import TestCase
import math
from util.decorators import timeit


@timeit
def procedure() -> float:
    sum = 0.0
    for index in range(10000):
        sum += math.exp(-index)
    return sum


class TestDecorator(TestCase):

    def test_timeit(self):
        print(procedure())
