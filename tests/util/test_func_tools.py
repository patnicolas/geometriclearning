from functools import total_ordering
from unittest import TestCase
import unittest
import logging
import os
import python
from python import SKIP_REASON

@total_ordering
class Student:
    def __init__(self, name: str, grade: float):
        self.name = name
        self.grade = grade

    def _is_valid_operand(self, that):
        return hasattr(that, "grade")

    def __gt__(self, that):
        if self._is_valid_operand(that):
            return NotImplemented
        else:
            return self.grade > that.grade

    def __eq__(self, that):
        if self._is_valid_operand(that):
            return NotImplemented
        else:
            return self.grade == that.grade


class TestFuncTools(TestCase):
    @unittest.skip("Not needed")
    def test_total_ordering(self):
        student1 = Student("Greg", 4.6)
        student2 = Student("Alice", 4.6)
        compared = Student("Greg", 4.6)>Student("Alice", 4.3)
        logging.info(compared)

    def test_partial(self):
        from functools import partial

        def adder(x: float, y:float) -> float:
            return x + y

        adder_10 = partial(adder, y = 10)
        adder_0 = partial(adder, y = 0)
        logging.info(f'Added to 10 {adder_10(5)}')
        logging.info(f'Adder keywords: {adder_10.keywords}')
        logging.info(f'Adder arguments: {adder_10.args}')


