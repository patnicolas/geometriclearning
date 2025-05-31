import os

from unittest import TestCase
import unittest
from util.http_post import HttpPost
import logging
import util


logging.info('path: ' + os.getcwd())


class TestHttpPost(TestCase):
    # @unittest.skip("Not needed")
    def test_process_predict(self):
        num_iterations = 1
        in_file = TestHttpPost.__predict_file()
        new_headers = {'Content-type': 'application/json', 'X-API-key':'ec18a88bc96d4c84bd4d6a0c4c8886ed'}
        post = HttpPost('predict_local', new_headers, True)
        successes, total_count = post.post_batch(in_file, False, num_iterations)
        logging.info(f'{successes=}, {total_count=}')

    @unittest.skip("Not needed")
    def test_batch_predict(self):
        in_file = TestHttpPost.__predict_file()
        new_headers = {'Content-type': 'application/json'}
        num_clients = 3
        for idx in range(num_clients):
            post = HttpPost('predict_training', new_headers, True)
            successes, total_count = post.post_batch(in_file, False, 1)
            logging.info(f'{successes=}, {total_count=}')

    @unittest.skip("Not needed")
    def test_process_feedback(self):
        in_file = TestHttpPost.__feedback_file()
        num_iterations = 4
        new_headers = {'Content-type': 'application/json'}
        post = HttpPost('feedback_local', new_headers, True)
        successes, total_count = post.post_batch(in_file, False, num_iterations)
        logging.info(f'{successes=}, {total_count=}')

    @staticmethod
    def __virtual_coder_file() -> str:
        in_file = "../../data/requests/ckr-diagnostic-request-stage.json"
        return in_file

    @staticmethod
    def __predict_file()-> str:
        in_file = "data/requests/streamline.json"
        return in_file

    @staticmethod
    def __feedback_file() -> str:
        in_file = "data/feedbacks/paris.json"
        return in_file