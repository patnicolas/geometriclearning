
import pandas as pd
import pygwalker as pyg
from typing import AnyStr


class GWalker(object):
    def __init__(self, csv_file: AnyStr):
        df = pd.read_csv(csv_file)
        self.walker = pyg.walk(df)


if __name__ == '__main__':
    my_csv_file = '../input/items.csv'
    gwalker = GWalker(my_csv_file)
