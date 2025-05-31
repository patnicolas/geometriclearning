from unittest import TestCase

import torch
from util.perf_eval import PerfEval
import logging
import util

def perf_test_func():
    x = torch.rand(200000)
    for index in range(100000):
        x = torch.exp(x * 0.9)
        x = x * 1.34

def acc_list1(lst: list) -> int:
    s = 0
    for n in lst:
        s += n
    return n

def acc_list2(lst: list) -> int:
    return sum(lst)


class TestPerfEval(TestCase):
    def test_eval(self):
        try:
            eval_perf = PerfEval(perf_test_func, None)
            eval_perf.eval()
        except Exception as e:
            logging.info(str(e))
            self.fail()

    def test_eval_acc1(self):
        try:
            lst = range(2, 100)
            eval_perf = PerfEval(acc_list1, lst)
            eval_perf.eval()
        except Exception as e:
            logging.info(str(e))
            self.fail()

    def test_eval_acc2(self):
        try:
            lst = range(2, 100)
            eval_perf = PerfEval(acc_list2, lst)
            eval_perf.eval()
        except Exception as e:
            logging.info(str(e))
            self.fail()