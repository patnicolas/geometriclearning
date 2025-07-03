__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import AnyStr, Self, Optional
from dl.model.neural_model import NeuralModel
from dl.block.variational_block import VariationalBlock
from dl import ConvException, MLPException, VAEException
import torch
import torch.nn as nn
import logging
import python
__all__ = ['VAEModel']

class VAEModel(NeuralModel):
    """
    Variational Autoencoder build from reusable encoders and dimension of the latent space.
    The Neural model can be a feedforward neural network, convolutional neural network....
    """
    def __init__(self,
                 model_id: AnyStr,
                 encoder: NeuralModel,
                 latent_dim: int,
                 decoder_out_activation: Optional[nn.Module] = None) -> None:
        """
        Constructor for the variational neural network
        @param model_id: Identifier for this model
        @type model_id: str
        @param encoder: Neural network encoder
        @type encoder: NeuralModel
        @param latent_dim: Size of the latent space
        @type latent_dim: int
        """
        try:
            assert latent_dim > 1, f'Dimension of latent space {latent_dim} should be > 1'

            # Create a decoder as inverted from the encoder (i.e. Deconvolution)
            decoder = encoder.transpose(decoder_out_activation) if decoder_out_activation is not None \
                else encoder.transpose()

            # extract the Torch modules
            modules = list(encoder.modules_seq.modules())

            # Build the inversion for the de convolutional network
            inverted_modules = list(decoder.modules_seq.modules())

            # construct the variational layer
            flatten_variational_input = encoder.get_flatten_output_size()
            variational_block = VariationalBlock.build(flatten_variational_input, latent_dim)

            # Build the Torch sequential module
            all_modules = modules + list(variational_block.modules) + inverted_modules
            modules_seq: torch.nn.Module = torch.nn.Sequential(*all_modules)

            # Call base class
            super(VAEModel, self).__init__(model_id, modules_seq)
            # Initialize the internal model parameters
            self.encoder = encoder
            self.decoder = decoder
            self.variational_block = variational_block
            # Used to collect distribution/tensor of the latent space
            self.z = None
        except ConvException as e:
            logging.error(str(e))
            raise VAEException((str(e)))
        except MLPException as e:
            logging.error(str(e))
            raise VAEException((str(e)))

    def get_mu_log_var(self) -> (nn.Module, nn.Module):
        return self.variational_block.mu, self.variational_block.log_var

    def __repr__(self) -> AnyStr:
        index2 = len(self.encoder.get_modules())
        index3 = index2+3
        return (self.encoder.list_modules(0) + self.variational_block.list_modules(index2) +
                f'\n{self.decoder.list_modules(index3)}')

    def __str__(self) -> AnyStr:
        return f'Model id: {self.model_id}\n*Encoder:{repr(self.encoder.modules)}' \
               f'\n*Variational: {repr(self.variational_block)}' \
               f'\n*Decoder: {repr(self.decoder.modules)}'

    def get_in_features(self) -> int:
        """
        Polymorphic method to retrieve the number of input features to the variational autoencoder
        @return: Number of input features
        @rtype: int
        """
        return self.encoder.get_in_features()

    def get_latent_features(self) -> int:
        return self.variational_block.sampler.in_features

    def get_out_features(self) -> int:
        """
        Polymorphic method to retrieve the number of output features to the variational autoencoder
        that are the same as the number of input features
        @return: Number of input features
        @rtype: int
        """
        self.encoder.get_conv_layer_out_shape()
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
        logging.info(x, 'Input dff_vae')
        x = self.encoder(x)
        logging.info(x, 'after encoder_model')
        original_shape = x.shape
        x = x.view(x.size(0), -1)
        logging.info(x, 'flattened')
        z, mu, log_var = self.variational_block(x)
        logging.info(z, 'after variational')
        self.z = z
        z = z.view(original_shape)
        logging.info(z, 'z.view')
        z = self.decoder(z)
        return z

    def transpose(self, extra: Optional[nn.Module] = None) -> Self:
        """
        Variational autoencoder is composed of an encoder and mirror decoder but cannot itself be inverted
        It throws a NotImplemented error
        """
        raise NotImplementedError('Cannot invert an Variational Autoencoder model')

    def save(self, extra_params: dict = None):
        raise NotImplementedError('NeuralModel.save is an abstract method')