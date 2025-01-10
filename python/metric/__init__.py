__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

class MetricException(Exception):
    def __init__(self, *args, **kwargs):
        super(MetricException, self).__init__(args, kwargs)
