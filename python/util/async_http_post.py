__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import asyncio
import requests
import urllib3

"""
    Implement Asynchronous/concurrent processing of a list of requests contained in an input file. 
    Once loaded, the requests are distributed acroass num_client threads to be processed concurrently
    against a service located in a give URL
    @param url: URL for the target service to process the requests
    @param headers: Dictionary of requests header parameters
    @param input_file: Input file containing the requests, one request per line
    @param num_clients: Number of concurrent clients
"""


class AsyncHttpPost(object):
    def __init__(self, url, headers, input_file: str, num_clients):
        assert 0 < num_clients < 50, f'AsyncHttpPost: num clients ${num_clients} should be ]0, 50['
        self.url = url
        self.headers = headers
        with open(input_file) as input:
            requests =  input.readlines()
            self.num_requests = len(requests)
            stride = self.num_requests // num_clients
            self.requests_iters = [iter(requests[i:i + stride]) for i in range(0, self.num_requests, stride)]

    async def post(self) -> list:
        """
            Process the list of iterator (1 iterator per client). The steps are
            1. Create a tasks from co-routine
            2. Aggregate the various tasks
            3. Block on the completion of all the tasks
            @return: List of results
        """
        tasks = self.__create_tasks()        # Step 1
        all_tasks = asyncio.gather(*tasks)   # Step 2
        responses = await all_tasks          # Step 3

        assert self.num_requests == len(responses), \
            f'Number of responses {len(responses)} != number of requests {self.num_requests}'
        return responses


    async def post_iter(self, requests_iter: iter, client_id: int) -> list:
        """
            Process an iterator of requests. The iterator is associated to a given client, 'client_id'
            @param requests_iter: Iterator of the requests to be processed by this client
            @param client_id: Simple identifier for this client
            @return: List of responses
        """
        constants.log_info(f'Enter {client_id}')
        responses = []
        while True:
            next_request = next(requests_iter, None)
            await asyncio.sleep(0)
            if next_request is None:
                break
            response = self.post_single(next_request)
            if not response:
                responses.append(response)
            else:
                constants.log_error(f'Empty response')
        constants.log_info(f'Exit {client_id}')
        return responses


    def post_single(self, request: str) -> str:
        """
            Execute a single post for a give request.
            @param request: Request for the post
            @return: Response
        """
        try:
            response = requests.post(self.url, data=request, headers=self.headers)
            constants.log_info("Received ML Response: {}".format(response.status_code))
            if response.status_code == 200:
                output = response.json()
            else:
                constants.log_error(f'Error: {request}')
                output = ""
        except urllib3.exceptions.ProtocolError as e:
            constants.log_error(str(e))
            output = ""
        except Exception as e:
            constants.log_error(str(e))
            output =  ""
        return output

    def __create_tasks(self):
        for idx, request_iter in enumerate(self.requests_iters):
            yield asyncio.create_task(self.post_iter(request_iter, idx))



async def execute_request(url: str, new_headers: dict, in_file: str, num_clients: int) -> list:
    """
        Wrapper around the processing of requests using concurrent client threads.
        @param url: URL for the target service to process the requests
        @param headers: Dictionary of requests header parameters
        @param input_file: Input file containing the requests, one request per line
        @param num_clients: Number of concurrent clients
        @return: List of responses should be same as number of requests
    """
    async_http_post = AsyncHttpPost(url, new_headers, in_file, num_clients)
    responses = await async_http_post.post()
    return responses