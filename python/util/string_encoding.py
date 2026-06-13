__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

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

__all__ = ['StringEncoding']


class StringEncoding(object):
    """
        Helper class to manipulate string, characters and bytes
    """
    @staticmethod
    def to_str(bytes_or_str)-> str:
        """
            Convert if a sequence of byte to a string
            @param bytes_or_str: A string or a sequence of bytes
            @return: A string
        """
        return bytes_or_str.decode('utf-8') if isinstance(bytes_or_str, bytes) else bytes_or_str

    @staticmethod
    def to_bytes(bytes_or_str) -> list:
        """
            Convert a string to a sequence of byte. The original sequence of bytes is returned
            @param bytes_or_str: A string or a sequence of bytes
            @return: A sequence of bytes
        """
        return bytes_or_str.encode('utf-8') if isinstance(bytes_or_str, str) else bytes_or_str