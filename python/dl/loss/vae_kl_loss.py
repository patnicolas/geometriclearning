__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from torch.nn.modules.loss import _Loss
import torch
from typing import AnyStr


class VAEKLLoss(_Loss):
    def __init__(self, mu: torch.Tensor, log_var: torch.Tensor, num_records: int):
        """
        Constructor for the Kullback-Leibler divergence based loss for variational auto-encoder
        @param mu: Mean for the normal distribution in the representation layer
        @type mu: Tensor
        @param log_var: Logarithm of the variance for the normal distribution in the representation layer
        @type log_var: Tensor
        @param num_records: Number of records
        @type num_records: int
        """
        super(VAEKLLoss, self).__init__(size_average=None, reduce=None, reduction='mean')
        self.mu = mu
        self.log_var = log_var
        self.num_records = num_records

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Implementation of the computation of the loss function for the Variational Auto-encoder
        @param _input: Input or predicted value
        @type _input: Tensor
        @param target: Target or labeled value
        @type target: Tensor
        @return: Loss function as a Torch tensor
        @rtype: Tensor
        """
        reconstruction_loss = self.hyper_params.loss_function(_input, target)
        kl_loss = (-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp())) / self.num_records
        return reconstruction_loss + kl_loss

    def __repr__(self) -> AnyStr:
        return f'\nMean: {self.mu}\nLog Variance: {self.log_var}\nNum records: {self.num_records}'