__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import torch
from torch import nn
from torch.autograd import Variable
from dl.block.neural_block import NeuralBlock
from typing import Self, overload
from dl.dl_exception import DLException
from util import log_size


class VariationalBlock(NeuralBlock):
    def __init__(self,  hidden_dim: int, latent_size: int):
        """
        Constructor for the variational Neural block of a variational auto-encoder
        @param hidden_dim:  Number of hidden unit to the variational block
        @type hidden_dim:  int
        @param latent_size:  Number of hidden unit for the latent space the variational block
        @type latent_size:  int
        """
        mu: nn.Module = nn.Linear(hidden_dim, latent_size)
        log_var: nn.Module = nn.Linear(hidden_dim, latent_size)
        sampler_fc: nn.Module = nn.Linear(latent_size, hidden_dim)
        modules = tuple([mu, log_var, sampler_fc])
        super(VariationalBlock, self).__init__(block_id='Gaussian', modules=modules)

        self.mu = mu
        self.log_var = log_var
        self.sampler_fc = sampler_fc

    def invert(self) -> Self:
        raise DLException('Cannot invert variational Neural block')

    def in_features(self) -> int:
        return self.mu.in_features

    def __repr__(self):
        return f'\n      Id: {self.block_id}\n      Mean: {repr(self.mu)}\n      logvar: {repr(self.log_var)}\n' \
               f'      Sampler: {repr(self.sampler_fc)}'

    @classmethod
    def re_parameterize(cls, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        random sample the z-space using the mean and log variance
        @param mu: Mean of the distribution in the z-space (latent)
        @param log_var: Logarithm of the variance of the distribution in the z-space
        @return: Sampled data point from z-space
        """
        std = log_var.mul(0.5).exp_()
        std_dev = std.data.new(std.size()).normal_()
        eps = Variable(std_dev)
        return eps.mul_(std).add_(mu)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Process the model as sequence of modules, implicitly called by __call__
        @param x: Input input_tensor as flattened output input_tensor from the convolutional layers
        @type x: Torch tensor
        @return: z, mean and log variance input_tensor
        @rtype: torch tensor
        """
        print(f'fc variational input shape {x.shape}')
        mu = self.mu(x)
        print(f'mu variational shape {mu.shape}')
        log_var = self.log_var(x)
        z = VariationalBlock.re_parameterize(mu, log_var)
        print(f'z variational shape {z.shape}')
        return self.sampler_fc(z), mu, log_var

