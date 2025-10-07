__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2023  All rights reserved."


# 3rd Party imports
import cProfile
import pstats
from pstats import SortKey
import logging
import line_profiler
import memory_profiler
from memory_profiler import profile
__all__ = ['Profiler']


class Profiler(object):
    """
    Wrapper for profiling a Python script or function
    """
    def __init__(self, python_script: str) -> None:
        """
        Constructor for the profiler wrapper
        @param python_script: Python script or main()
        """
        self.python_script = python_script

    def run(self, num_records: int, stats_file_name: str) -> None:
        assert num_records > 1, f'Number of records for Profiler {num_records} should be  > 1'
        """
            Use C-Profiler to compute the time duration of a function and inner calls. The
            methods are sorted by decreasing order of cumulative time
            @param num_records: Number of methods with the highest cumulative time
            @param stats_file_name: Name of the file containing statistics
        """
        cProfile.run(self.python_script, stats_file_name)

        p = pstats.Stats(stats_file_name)
        p.sort_stats(SortKey.CUMULATIVE)
        p.print_stats(num_records)

    def run_line_profiler(self) -> None:
        """
        Execute the Kern - line profiler for a script or function defined in the constructor
        """
        ln_profiler = line_profiler.LineProfiler()
        ln_profiler.enable()
        ln_profiler.enable_by_count()
        lp_wrapper = ln_profiler(self.python_script)
        lp_wrapper()
        ln_profiler.print_stats()

    def run_memory_profiler_func(self) -> None:
        """
        Execute the memory line profiler on a script or Python function defined in the constructor
        """
        usage = memory_profiler.memory_usage(self.python_script)
        logging.info(f'Memory {usage=}')

    @staticmethod
    def run_memory_profiler() -> None:
        """
        Execute the memory line profiler on a script or Python function defined in the constructor
        """
        usage = memory_profiler.memory_usage(-1)
        logging.info(f'Memory {usage=}')


@profile
def test_func():
    import math
    arr = []
    for i in range(0, 100000):
        arr.append(math.sin(i*0.001) + math.log(1.0 + i*0.0002))
    del arr


if __name__ == '__main__':
    profiler = Profiler('test_func()')
    profiler.run(20, '../output/stats-results')