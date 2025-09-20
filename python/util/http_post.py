__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard Library imports
import logging
from typing import Dict, AnyStr
# 3rd Party imports
import requests
import time
import urllib3
# Library imports
from util.timeit import timeit
__all__ = ['HttpPost']


class HttpPost(object):
    """"
    Simple synchronous implementation of HTTP post using JSON input_tensor line or file. The target can be defined as
    - Fully defined URL (i.e. http://ip-10-5-47-158.us-east-2.compute.internal:8087/geminiml/predict)
    - Pre-defined targets (i.e. feedback_local)
    Command line
        python $target $is_predefined_target
    Example
        python stage_predicted True input_file
        python http://ip-10-5-47-158.us-east-2.compute.internal:8087/geminiml/predict False input_file
    """
    def __init__(self, target: str, headers: list, predefined_targets: Dict[AnyStr, AnyStr] = False) -> None:
        """
        Constructor for HttpPost
        @param target: URL (if is_predefined_target == True) or a pre-defined
        @param headers: Header of the HTTP Post
        @param predefined_targets: Specify if this is a predefined target (True) or a vision URL (False)
        @exception KeyError: Predefined URL is not found
        """
        self.headers = headers
        # If this is a predefined target.
        if len(predefined_targets) > 0:
            try:
                self.target = predefined_targets[target]
            except KeyError as e:
                self.target = None
                logging.error(str(e))
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
                logging.info("Received ML Response: {}".format(response.status_code))
                if response.status_code == 200:
                    json_response = response.json()
                    logging.info(f'{json_response=}')
                    return True
                else:
                    logging.error(f'Error: {line}')
                    return False
            except urllib3.exceptions.ProtocolError as e:
                logging.error(str(e))
                return False
            except Exception as e:
                logging.error(str(e))
                return False
        else:
            return False

    @timeit
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
                        logging.info(f'{success_count=}, {total_count=}')
                        time.sleep(sleep_time)
        return success_count, total_count


def main(args: list):
    new_headers = {'Content-type': 'application/json'}
    target = args[0]
    is_predefined_target = args[1]
    input_file = args[2]
    post = HttpPost(target, new_headers, is_predefined_target)
    successes, total = post.post_batch(input_file)


import sys

if __name__ == "__main__":
    main(sys.argv[1:])
