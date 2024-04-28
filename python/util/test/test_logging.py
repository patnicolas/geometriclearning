from unittest import TestCase

import constants


class Generators(object):
    @staticmethod
    def even(lst: list):
        return [i for i in lst if i % 2]


class TestLogger(TestCase):
    def test___init(self):
        lst = [1, 2, 3, 6, 7, 9, 10]
        [print(j) for j in Generators.even(lst)]

    def test_log(self):
        constants.log_error('hello')
