from unittest import TestCase
import unittest
from util.profiler import Profiler
import os
import python
from python import SKIP_REASON

def test_func():
    import math
    arr=[]
    for i in range(0, 100000):
        arr.append(math.sin(i*0.001) + math.log(1.0 + i*0.0002))


class TestProfiler(TestCase):
    def test_run(self):
        try:
            profiler = Profiler('main()')
            profiler.run(100000, '../../output/stats-results')
        except Exception as e:
            self.fail(str(e))

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_line_profiler(self):
        profiler = Profiler(test_func)
        profiler.run_line_profiler()

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_memory_profiler_func(self):
        profiler = Profiler(test_func)
        profiler.run_memory_profiler_func()

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_memory_profiler(self):
        profiler = Profiler(test_func)
        profiler.run_memory_profiler()


def main():
    test_func()
