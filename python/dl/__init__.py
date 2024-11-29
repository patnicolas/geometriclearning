__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."


class DLException(Exception):
    def __init__(self, *args, **kwargs):
        super(DLException, self).__init__(args, kwargs)


class ConvException(DLException):
    def __init__(self, *args, ** kwargs):
        super(ConvException, self).__init__(args, kwargs)


class VAEException(DLException):
    def __init__(self, *args, **kwargs):
        super(VAEException, self).__init__(args, kwargs)


class GraphException(DLException):
    def __init__(self, *args, **kwargs):
        super(GraphException, self).__init__(args, kwargs)