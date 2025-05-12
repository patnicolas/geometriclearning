__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

"""
    Helper class to manipulate string, characters and bytes
"""


class StringEncoding(object):
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