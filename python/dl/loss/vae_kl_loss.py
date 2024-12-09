__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
from typing import AnyStr, Optional

from dl import VAEException


class VAEKLLoss(_Loss):
    def __init__(self,
                 mu: nn.Module,
                 log_var: nn.Module,
                 num_records: int,
                 loss_func: nn.Module,
                 beta: Optional[float] = 1.0) -> None:
        """
        Constructor for the Kullback-Leibler divergence based loss for variational auto-encoder
            Total loss = reconstruction loss + beta* KL loss
        @param mu: Linear layer for the normal distribution in the representation layer
        @type mu: nn.Module
        @param log_var: Logarithm of the variance in the linear module the normal distribution in the representation layer
        @type log_var: nn.Module
        @param num_records: Number of records
        @type num_records: int
        @param loss_func: Loss function for the encoder
        @type loss_func: nn.Module
        @param beta: Optional beta parameter
        @type beta: float
        """
        super(VAEKLLoss, self).__init__(size_average=None, reduce=None, reduction='mean')
        self.mu = mu
        self.log_var = log_var
        self.num_records = num_records
        self.loss_func = loss_func
        self.beta = beta

    def forward(self, x: torch.Tensor, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Implementation of the computation of the loss function for the Variational Auto-encoder
        @param x: Input or predicted value
        @type x: Tensor
        @param z: Predicted latent value
        @type z: Tensor
        @param target: Target or labeled value
        @type target: Tensor
        @return: Loss function as a Torch Tensor
        @rtype: Tensor
        """
        if self.log_var is None or self.mu is None:
            raise VAEException(f'Log var and mu variational parameters are not defined')

        reconstruction_loss = self.loss_func(x, target)
        log_var_value = self.log_var(z)
        mu_value = self.mu(z)
        kl_loss = (-0.5 * torch.sum(1 + log_var_value - mu_value ** 2 - log_var_value.exp())) / self.num_records
        if kl_loss == torch.inf or kl_loss > 1e+6:
            kl_loss = 1e+6
        return reconstruction_loss + self.beta*kl_loss

    def __repr__(self) -> AnyStr:
        return (f'\nMean: {self.mu}\nLog Variance: {self.log_var}\nNum records: {self.num_records}'
                f'\nLoss function: {self.loss_func}\nBeta: {self.beta}')