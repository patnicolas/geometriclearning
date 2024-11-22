__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."


class ConvException(Exception):
    def __init__(self, *args, ** kwargs):
        super(ConvException, self).__init__(args, kwargs)


class DLException(Exception):
    def __init__(self, *args, **kwargs):
        super(DLException, self).__init__(args, kwargs)


class GraphException(Exception):
    def __init__(self, *args, **kwargs):
        super(GraphException, self).__init__(args, kwargs)