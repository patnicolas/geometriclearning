from unittest import TestCase
import logging
import python

class Generators(object):
    @staticmethod
    def even(lst: list):
        return [i for i in lst if i % 2]


class TestLogger(TestCase):
    def test___init(self):
        lst = [1, 2, 3, 6, 7, 9, 10]
        [logging.info(j) for j in Generators.even(lst)]

    def test_log(self):
        logging.error('hello')
