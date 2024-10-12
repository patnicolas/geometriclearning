__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2024  All rights reserved."


class MCMC(object):
    from abc import abstractmethod

    @abstractmethod
    def sample(self, theta: float) -> float:
        raise NotImplementedError('MCMC.sample is an abstract method')
