__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import torch
from torch import nn
from dl.block.neural_block import NeuralBlock
from typing import Self, AnyStr, Optional
from dl import VAEException


class VariationalBlock(NeuralBlock):
    def __init__(self,  hidden_dim: int, latent_size: int):
        """
        Constructor for the variational Neural block of a variational auto-encoder
        @param hidden_dim:  Number of hidden unit to the variational block
        @type hidden_dim:  int
        @param latent_size:  Number of hidden unit for the latent space the variational block
        @type latent_size:  int
        """
        mu: nn.Module = nn.Linear(in_features=hidden_dim, out_features=latent_size)
        log_var: nn.Module = nn.Linear(in_features=hidden_dim, out_features=latent_size)
        sampler_fc: nn.Module = nn.Linear(in_features=latent_size, out_features=hidden_dim)
        modules = [mu, log_var, sampler_fc]
        super(VariationalBlock, self).__init__(block_id='Variational', modules=tuple(modules))

        self.mu = mu
        self.log_var = log_var
        self.sampler_fc = sampler_fc

    def transpose(self, extra: Optional[nn.Module] = None) -> Self:
        raise VAEException('Cannot invert variational Neural block')

    def in_features(self) -> int:
        return self.mu.in_features

    def list_modules(self, index: int = 0) -> AnyStr:
        return (f'\n{index}: Mean-{str(self.mu)}\n{index+1}: LogVar-{str(self.log_var)}'
                f'\n{index+2}: Sampler-{str(self.sampler_fc)}')

    def __repr__(self) -> AnyStr:
        return f'\n      Id: {self.block_id}\n      Mean: {str(self.mu)}\n      logvar: {str(self.log_var)}\n' \
               f'      Sampler: {str(self.sampler_fc)}'

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
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Process the model as sequence of modules, implicitly called by __call__
        @param x: Input input_tensor as flattened output input_tensor from the convolutional layers
        @type x: Torch tensor
        @return: z, mean and log variance input_tensor
        @rtype: torch tensor
        """
        # print(f'fc variational input shape {x.shape}')
        mu = self.mu(x)
        # print(f'mu variational shape {mu.shape}')
        log_var = self.log_var(x)
        z = VariationalBlock.re_parameterize(mu, log_var)
        # print(f'z variational shape {z.shape}')
        return self.sampler_fc(z), mu, log_var

