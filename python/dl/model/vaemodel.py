__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, Self, overload
from dl.model.neuralmodel import NeuralModel
from dl.block.variationalblock import VariationalBlock
import torch

import logging
logger = logging.getLogger('dl.model.VAEModel')

"""
Variational Autoencoder build from reusable encoders and dimension of the latent space.
The Neural model can be a feedforward neural network, convolutional neural network....
"""


class VAEModel(NeuralModel):
    def __init__(self, model_id: AnyStr, encoder: NeuralModel, latent_size: int):
        """
        Constructor for the variational neural network
        @param model_id: Identifier for this model
        @type model_id: str
        @param encoder: Neural network encoder
        @type encoder: NeuralModel
        @param latent_size: Size of the latent space
        @type latent_size: int
        """
        # Create a decoder as inverted from the encoder
        decoder = encoder.invert()
        # extract the Torch modules
        modules = list(encoder.get_model().modules())
        inverted_modules = list(decoder.get_model().modules())
        # construct the variational layer
        variational_block = VariationalBlock(encoder.get_out_features(), latent_size)
        # Build the Torch sequential module
        all_modules = modules + variational_block.modules + inverted_modules
        seq_module: torch.nn.Module = torch.nn.Sequential(*all_modules)
        # Call base class
        super(VAEModel, self).__init__(model_id, seq_module)
        # Initialize the internal model parameters
        self.encoder = encoder
        self.decoder = decoder
        self.variational_block = variational_block
        self.mu = None
        self.log_var = None

    def __repr__(self):
        return f'Model id: {self.model_id}\n *Encoder:{repr(self.encoder)}' \
               f'\n *Variational: {repr(self.variational_block)}' \
               f'\n * Decoder: {repr(self.decoder)}'

    def get_in_features(self) -> int:
        """
        Polymorphic method to retrieve the number of input features to the variational autoencoder
        @return: Number of input features
        @rtype: int
        """
        return self.encoder.get_in_features()

    def get_latent_features(self) -> int:
        return self.variational_block.sampler_fc.in_features

    def get_out_features(self) -> int:
        """
        Polymorphic method to retrieve the number of output features to the variational autoencoder
        that are the same as the number of input features
        @return: Number of input features
        @rtype: int
        """
        return self.encoder.get_in_features()

    def latent_parameters(self) -> (torch.Tensor, torch.Tensor):
        """
        Return the latent parameters (mean and logarithm of variance)
        @return: Tuple of tensor representing mean and log of variance
        @rtype: Tuple[Tensor, Tensor]
        """
        return self.mu, self.log_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the model as sequence of modules, implicitly called by __call__. The mean and logarithm
        of the variance for the Gaussian distribution approximation in the latent space are also computed
        and stored in the instance variable, mu and log_var
        @param x: Input input_tensor
        @type x: Torch Tensor
        @return: z
        @rtype: Torch Tensor
        """
        logger.info(x, 'Input dff_vae')
        x = self.encoder(x)
        logger.info(x, 'after encoder_model')
        batch, a = x.shape
        x = x.view(batch, -1)
        logger.info(x, 'flattened')
        z, mu, log_var = self.variational_block(x)
        logger.info(z, 'after variational')
        z = z.view(batch, a)
        logger.info(z, 'z.view')
        z = self.decoder(z)
        self.mu = mu
        self.log_var = log_var
        return z

    def invert(self) -> Self:
        """
        Variational autoencoder is composed of an encoder and mirror decoder but cannot itself be inverted
        It throws a NotImplemented error
        """
        raise NotImplementedError('Cannot invert an Variational Autoencoder model')

    def save(self, extra_params: dict = None):
        raise NotImplementedError('NeuralModel.save is an abstract method')