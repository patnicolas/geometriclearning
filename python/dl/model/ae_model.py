__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr, Self
from dl.model.neural_model import NeuralModel
import torch
import logging
logger = logging.getLogger('dl.model.AEModel')

"""
Define a 'plain vanilla' auto-encoder as an encoder and decoder feed forward neural network. 
The encoder is provided as the class argument and the decoder is automatically generated 
through inversion of the neural blocks it contains.
"""


class AEModel(NeuralModel):

    def __init__(self, model_id: AnyStr, encoder: NeuralModel) -> None:
        """
        Constructor
        @param model_id: Identifier for this Auto-encoder
        @type model_id: AnyStr
        @param encoder: Encoder neural model
        @type encoder: NeuralModel
        """
        _decoder = encoder.invert()

        modules = list(encoder.get_model().modules())
        inverted_modules = list(_decoder.get_model().modules())
        all_modules = modules + inverted_modules
        seq_module: torch.nn.Module = torch.nn.Sequential(*all_modules)
        super(AEModel, self).__init__(model_id, seq_module)
        self.encoder = encoder
        self.decoder = _decoder

    def __repr__(self):
        return f'Model id: {self.model_id}\n *Encoder:  {repr(self.encoder)}\n *Decoder:  {repr(self.decoder)}'

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

    def invert(self) -> Self:
        """
        Autoencoder is composed of an encoder and mirror decoder but cannot itself be inverted
        It throws a NotImplemented error
        """
        raise NotImplementedError('Cannot invert an Autoencoder model')

    def save(self, extra_params: dict = None):
        raise NotImplementedError('NeuralModel.save is an abstract method')
