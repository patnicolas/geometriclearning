__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
from torch import nn
from dl.block.neural_block import NeuralBlock
from typing import Self, AnyStr, Optional
from dl import VAEException


class VariationalBlock(NeuralBlock):
    def __init__(self,  mu: nn.Linear, log_var: nn.Linear, sampler: nn.Linear):
        """
        Constructor for the variational Neural block of a variational auto-encoder
        @param mu: Linear module for the mean of the Gaussian distribution
        @type mu: nn.Linear
        @param log_var: Linear module for the log of the variance of the Gaussian distribution
        @type log_var: nn.Linear
        @param sampler: Linear module for sampling the learned Gaussian distribution (Mean, variance)
        @type sampler: nn.Linear

        """
        self.mu = mu
        self.log_var = log_var
        self.sampler = sampler
        super(VariationalBlock, self).__init__(block_id='Variational',
                                               modules=(mu, log_var, sampler))

    @staticmethod
    def build(cls, hidden_dim: int, latent_size: int) -> Self:
        """
        Constructor for the variational Neural block of a variational auto-encoder
        @param hidden_dim:  Number of hidden unit to the variational block
        @type hidden_dim:  int
        @param latent_size:  Number of hidden unit for the latent space the variational block
        @type latent_size:  int
        """
        mu = nn.Linear(in_features=hidden_dim, out_features=latent_size, bias=True)
        log_var = nn.Linear(in_features=hidden_dim, out_features=latent_size, bias=True)
        sampler = nn.Linear(in_features=latent_size, out_features=hidden_dim, bias=True)
        return cls(mu, log_var, sampler)

    def transpose(self, extra: Optional[nn.Module] = None) -> Self:
        raise VAEException('Cannot invert variational Neural block')

    def in_features(self) -> int:
        return self.mu.in_features

    def list_modules(self, index: int = 0) -> AnyStr:
        return (f'\n{index}: Mean-{str(self.mu)}\n{index+1}: LogVar-{str(self.log_var)}'
                f'\n{index+2}: Sampler-{str(self.sampler)}')

    def __repr__(self) -> AnyStr:
        return f'\n      Id: {self.block_id}\n      Mean: {str(self.mu)}\n      logvar: {str(self.log_var)}\n' \
               f'      Sampler: {str(self.sampler)}'

    @classmethod
    def re_parameterize(cls, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        random sample the z-space using the mean and log variance
        @param mu: Mean of the distribution in the z-space (latent)
        @type mu: torch.Tensor
        @param log_var: Logarithm of the variance of the distribution in the z-space
        @type log_var: torch.Tensor
        @return: Sampled data point from z-space
        """
        # Compute standard deviation from logvar
        std = torch.exp(0.5 * log_var)
        # Sample epsilon from N(0, 1)
        eps = torch.randn_like(std)
        # Re-parameterization trick
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Process the model as sequence of modules, implicitly called by __call__
        @param x: Input input_tensor as flattened output input_tensor from the convolutional layers
        @type x: Torch tensor
        @return: z, mean and log variance input_tensor
        @rtype: torch tensor
        """
        # Compute the mean for the tensor
        x = VariationalBlock.__laplace_zero_tensor(x)
        mu = self.mu(x)
        # Computes log (exp(sqr(std))
        log_var = self.log_var(x)
        z = VariationalBlock.re_parameterize(mu, log_var)
        if VariationalBlock.__is_z_nan(z):
            has_non_zeros = VariationalBlock.input_has_non_zeros(x)
            raise VAEException(f'VAE z is nan with x has non zeros? {has_non_zeros}')
        return self.sampler(z), mu, log_var

    @staticmethod
    def __is_z_nan(z: torch.Tensor) -> bool:
        dim = len(z.shape)
        return (dim == 1 and torch.isnan(z[0])) or (dim == 2 and torch.isnan(z[0][0]))

    @staticmethod
    def __laplace_zero_tensor(x: torch.Tensor) -> torch.Tensor:
        if not bool(x.any()):
            s = list(x.shape)
            return (torch.rand(s) - 0.5).to(torch.device("mps"))
        else:
            return x




