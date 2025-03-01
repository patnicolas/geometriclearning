__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from typing import AnyStr, Self, Any
from dl.model.neural_model import NeuralModel
import torch.nn as nn
import torch
import logging
logger = logging.getLogger('dl.model.AEModel')

"""
Define a 'plain vanilla' auto-encoder as an encoder and decoder feed forward neural network. 
The encoder is provided as the class argument and the decoder is automatically generated 
through inversion of the neural blocks it contains.
"""


class AEModel(NeuralModel):

    def __init__(self,
                 model_id: AnyStr,
                 encoder: NeuralModel,
                 latent_block: Any = None,
                 decoder_out_activation: nn.Module = None) -> None:
        """
        Constructor
        @param model_id: Identifier for this Auto-encoder
        @type model_id: AnyStr
        @param encoder: Encoder neural model
        @type encoder: NeuralModel
        """
        decoder = encoder.transpose(decoder_out_activation) if decoder_out_activation is not None \
            else encoder.transpose()

        self.encoder = encoder
        self.decoder = decoder

        modules = list(encoder.get_model().modules())
        inverted_modules = list(decoder.get_model().modules())
        all_modules = modules + inverted_modules if latent_block is None \
            else  modules + latent_block+ inverted_modules
        seq_module = torch.nn.Sequential(*all_modules)
        super(AEModel, self).__init__(model_id, seq_module)

    def __str__(self):
        return f'Model id: {self.model_id}\n *Encoder:  {str(self.encoder)}\n *Decoder:  {str(self.decoder)}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the model as sequence of modules, implicitly called by __call__.
        @param x: Input input_tensor
        @type x: Torch Tensor
        @return: z
        @rtype: Torch Tensor
        """
        logger.info(x, 'Input dff_vae')
        x = self.encoder_model(x)
        logger.info(x, 'after encoder_model')
        return self.decoder_model(x)

    def get_latent_features(self) -> int:
        return self.encoder.get_out_features()

    def get_in_features(self) -> int:
        return self.encoder.get_in_features()

    def get_out_features(self) -> int:
        """
        Polymorphic method to retrieve the number of output features. It should be the same as
        the number of input features
        @return: Number of input features
        @rtype: int
        """
        return self.encoder.get_in_features()

    def transpose(self) -> Self:
        """
        Autoencoder is composed of an encoder and mirror decoder but cannot itself be inverted
        It throws a NotImplemented error
        """
        raise NotImplementedError('Cannot invert an Autoencoder model')

    def save(self, extra_params: dict = None):
        raise NotImplementedError('NeuralModel.save is an abstract method')
