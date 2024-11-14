__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import requests
import time
import urllib3
from util.decorators import timeit

""""
	Simple synchronous implementation of HTTP post using JSON input_tensor line or file. The target can be defined as 
	- Fully defined URL (i.e. http://ip-10-5-47-158.us-east-2.compute.internal:8087/geminiml/predict)
	- Pre-defined targets (i.e. feedback_local)
	Command line
		python $target $is_predefined_target
	Example
		python stage_predicted True input_file
		python http://ip-10-5-47-158.us-east-2.compute.internal:8087/geminiml/predict False input_file
	
	@param target: URL (if is_predefined_target == True) or a pre-defined
	@param headers: Header of the HTTP Post
	@param is_predefined_target: Specify if this is a predefined target (True) or a vision URL (False)
	:exception KeyError: Predefined URL is not found
"""


class HttpPost(object):
    def __init__(self, target: str, headers: list, is_predefined_target: bool = False):
        self.headers = headers
        # If this is a predefined target.
        if is_predefined_target:
            try:
                self.target = predefined_targets[target]
            except KeyError as e:
                self.target = None
                constants.log_error(str(e))
        # Otherwise a URL
        else:
            self.target = target

    def post_single(self, line: str) -> bool:
        """
			Process an doc_terms_str as single JSON line
			@param line: A single JSON structure
			@return: True if request successful, False otherwise
		"""
        if self.target is not None:
            try:
                response = requests.post(self.target, data=line, headers=self.headers)
                constants.log_info("Received ML Response: {}".format(response.status_code))
                if response.status_code == 200:
                    json_response = response.json()
                    constants.log_info(json_response)
                    return True
                else:
                    constants.log_error(f'Error: {line}')
                    return False
            except urllib3.exceptions.ProtocolError as e:
                constants.log_error(str(e))
                return False
            except Exception as e:
                constants.log_error(str(e))
                return False
            except TypeError as e:
                constants.log_error(str(e))
                return False
        else:
            return False

    # @timeit
    def post_batch(
            self,
            input_file: str,
            single_record: bool,
            num_iterations: int = 1,
            sleep_time: float = 0.2) -> (int, int):
        """
            Process a batch of JSON entries defined within a single file
            @param input_file: Input file containing the JSON formatted requests
            @param single_record: The file contains a single records
            @param num_iterations: Number of iterations (default 1)
            @param sleep_time: Number of seconds (fraction) the HTTP client thread pauses
            @return: Pair Number of success requests, Total number of requests
        """
        # start_time = time.time()
        total_count = 0
        success_count = 0
        with open(input_file) as input:
            if single_record:
                content = input.read()
                all_requests = [content.replace('\n', ' ')]
            else:
                all_requests = input.readlines()

            if all_requests:
                for _ in range(num_iterations):
                    for line in all_requests:
                        if self.post_single(line):
                            success_count += 1
                        total_count += 1
                        print(f'Successes: {success_count} Count: {total_count}')
                        time.sleep(sleep_time)
        return success_count, total_count


predefined_targets = {
    'predict_local': 'http://127.0.0.1:8080/geminiml/predict',
    'predict_test': 'http://ip-10-5-58-7.us-east-2.compute.internal:8087/geminiml/predict',
    'predict_stage': 'http://ip-10-5-45-112.us-east-2.compute.internal:8087/geminiml/predict',
    'predict_production': 'http://ip-10-5-59-20.us-east-2.compute.internal:8087/geminiml/predict',
    'predict_training': 'http://ip-10-5-38-237.us-east-2.compute.internal:8087/geminiml/predict',
    'predict_demo': 'http://ip-10-5-53-143.us-east-2.compute.internal:8087/geminiml/predict',
    'predict_new_api': 'http://ip-10-5-33-98.us-east-2.compute.internal:8087/geminiml/predict',
    'feedback_local': 'http://localhost:8080/geminiml/feedback',
    'feedback_test': 'http://ip-10-5-58-7.us-east-2.compute.internal:8087/geminiml/feedback',
    'feedback_stage': 'http://ip-10-5-45-112.us-east-2.compute.internal:8087/geminiml/feedback',
    'feedback_production': 'http://ip-10-5-59-20.us-east-2.compute.internal:8087/geminiml/feedback',
    'feedback_training': 'http://ip-10-5-38-237.us-east-2.compute.internal:8087/geminiml/feedback',
    'virtual_coder_stage': 'http://vc-stage.private.aideo-tech.com/api',
    'virtual_coder_production': 'https://virtual-coder.aideo-tech.com/api'
}


def main(args: list):
    new_headers = {'Content-type': 'application/json'}
    target = args[0]
    is_predefined_target = args[1]
    input_file = args[2]
    post = HttpPost(target, new_headers, is_predefined_target)
    successes, total = post.post_batch(input_file)
    constants.log_info(f'Successes: {successes} All counts {total}')


import sys

if __name__ == "__main__":
    main(sys.argv[1:])
