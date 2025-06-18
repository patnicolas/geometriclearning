import unittest
from unittest import TestCase
from util.io_util import IOUtil
import logging
import os
import python
from python import SKIP_REASON


class TestIOUtil(TestCase):
    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_to_json(self):
        try:
            file_name = '../../data/test.json'
            ioutil = IOUtil(file_name)
            json_content = ioutil.to_json()
            logging.info(json_content)
        except Exception as e:
            self.fail(str(e))

    def test_pickle(self):
        try:
            file_name = '../../input/test1-pickle'
            my_dict = {"a": 1, "b": 2}
            lst = [my_dict, my_dict]
            ioutil = IOUtil(file_name)
            ioutil.to_pickle(lst)
            new_lst = ioutil.from_pickle()
            logging.info(f'{new_lst=}')
        except Exception as e:
            self.fail(str(e))

