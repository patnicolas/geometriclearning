__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, Self
from python.dl.model.neuralmodel import NeuralModel
from python.dl.block.variationalblock import VariationalBlock
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


"""
import torch
from torch import nn
from torch.nn import functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(VariationalAutoencoder, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_mean = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim2)
        
        # Decoder

        self.fc4 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc5 = nn.Linear(hidden_dim1, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_mean(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        return torch.sigmoid(self.fc5(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Hyperparameters
input_dim = 784  # For example, flattened 28x28 images from MNIST
hidden_dim1 = 400
hidden_dim2 = 200
latent_dim = 20

# Model
model = VariationalAutoencoder(input_dim, hidden_dim1, hidden_dim2, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Example of a training step
def train_step(model, data):
    model.train()
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(data)
    loss = loss_function(recon_batch, data, mu, logvar)
    loss.backward()
    optimizer.step()
    return loss.item()

# Assuming 'data' is a batch of input data
# loss = train_step(model, data)




"""
