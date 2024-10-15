

class ConvException(Exception):
    def __init__(self, *args, ** kwargs):
        super(ConvException, self).__init__(args, kwargs)