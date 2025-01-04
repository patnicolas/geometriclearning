

import numpy as np


class EinSum(object):
    def __init__(self, first: np.array, second: np.array= None) -> None:
        self.first = first
        self.second = second

    def dot(self) -> np.array:
        assert self.second is not None and \
                len(self.first.shape) == len(self.second.shape) == 1 and \
                self.first.shape[0] == self.first.shape[0]
        return np.einsum('i,i->', self.first, self.second)

    def matrix_mul(self) -> np.array:
        assert  self.second is not None and \
                len(self.first.shape) == 2 \
                and self.first.shape == self.second.shape
        return np.einsum('ij,jk->ik', self.first, self.second)

    def matrix_el_sum(self) -> np.array:
        assert  self.second is not None and \
                len(self.first.shape) == 2 \
                and self.first.shape == self.second.shape
        return np.einsum('ij,ij->', self.first, self.second)

    def outer_product(self) -> np.array:
        assert self.second is not None
        return np.einsum('i,j->ij', self.first, self.second)

    def transpose(self) -> np.array:
        assert len(self.first.shape) == 2
        return np.einsum('ij->ji', self.first)

    def trace(self) -> np.array:
        assert len(self.first.shape) == 2
        return np.einsum('ii->', self.first)

