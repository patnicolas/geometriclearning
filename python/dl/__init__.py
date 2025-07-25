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

__all__ = ['MLPException', 'ConvException', 'VAEException', 'GraphException', 'TrainingException', 'ValidationException']

class MLPException(Exception):
    def __init__(self, *args, **kwargs):
        super(MLPException, self).__init__(args, kwargs)


class ConvException(Exception):
    def __init__(self, *args, ** kwargs):
        super(ConvException, self).__init__(args, kwargs)


class GraphException(Exception):
    def __init__(self, *args, ** kwargs):
        super(GraphException, self).__init__(args, kwargs)


class VAEException(Exception):
    def __init__(self, *args, **kwargs):
        super(VAEException, self).__init__(args, kwargs)


class TrainingException(MLPException):
    def __init__(self, *args, **kwargs):
        super(TrainingException, self).__init__(args, kwargs)


class ValidationException(MLPException):
    def __init__(self, *args, **kwargs):
        super(ValidationException, self).__init__(args, kwargs)
        
