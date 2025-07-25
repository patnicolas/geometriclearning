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

from typing import AnyStr, Self, Any
from dl.model.neural_model import NeuralModel
import torch.nn as nn
import torch
import logging
import python
__all__ = ['AEModel']


class AEModel(NeuralModel):
    """
    Define a 'plain vanilla' auto-encoder as an encoder and decoder feed forward neural network.
    The encoder is provided as the class argument and the decoder is automatically generated
    through inversion of the neural blocks it contains.
    """
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
        logging.debug(x, 'Input dff_vae')
        x = self.encoder_model(x)
        logging.debug(x, 'after encoder_model')
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
