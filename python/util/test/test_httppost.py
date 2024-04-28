import os

from unittest import TestCase
import unittest
import constants
from util.httppost import HttpPost


print('path: ' + os.getcwd())


class TestHttpPost(TestCase):
    # @unittest.skip("Not needed")
    def test_process_predict(self):
        num_iterations = 1
        in_file = TestHttpPost.__predict_file()
        new_headers = {'Content-type': 'application/json', 'X-API-key':'ec18a88bc96d4c84bd4d6a0c4c8886ed'}
        post = HttpPost('predict_local', new_headers, True)
        successes, total = post.post_batch(in_file, False, num_iterations)
        constants.log_info(f'Success: {successes} All counts {total}')

    @unittest.skip("Not needed")
    def test_batch_predict(self):
        in_file = TestHttpPost.__predict_file()
        new_headers = {'Content-type': 'application/json'}
        num_clients = 3
        for idx in range(num_clients):
            post = HttpPost('predict_training', new_headers, True)
            successes, total = post.post_batch(in_file, False, 1)
            constants.log_info(f'Success: {successes} All counts {total}')

    @unittest.skip("Not needed")
    def test_process_feedback(self):
        in_file = TestHttpPost.__feedback_file()
        num_iterations = 4
        new_headers = {'Content-type': 'application/json'}
        post = HttpPost('feedback_local', new_headers, True)
        successes, total = post.post_batch(in_file, False, num_iterations)
        constants.log_info(f'Success: {successes} All counts {total}')

    @staticmethod
    def __virtual_coder_file() -> str:
        in_file = "../../data/requests/ckr-diagnostic-request-stage.json"
        return in_file

    @staticmethod
    def __predict_file()-> str:
        # in_file = "data/requests/cmbs.json"
        # in_file = "data/requests/brault.json"
        # in_file = "data/requests/rpa.json"
        # in_file = "data/requests/surgical-notes.json"
        # in_file = "data/requests/vc-failure.json"
        # in_file = "data/requests/mix.json"
        # in_file = "data/requests/paris.json"
        in_file = "data/requests/streamline.json"
        # in_file = "data/requests/mix-specialties.json"
        # in_file =  "data/requests/utrad.json"
        # in_file = "data/requests/mix-load.json"
        # in_file = "data/requests/test1-production.json"

        return in_file

    @staticmethod
    def __feedback_file() -> str:
        in_file = "data/feedbacks/paris.json"
        # in_file = "data/feedbacks/feedbackRequest.json"
        # in_file = "data/feedbacks/brault.json"
        #  in_file = "data/feedbacks/utrad.json"
        # n_file = "data/feedbacks/mix-load.json"
        # in_file = "data/feedbacks/mix-test1.json"
        return in_file