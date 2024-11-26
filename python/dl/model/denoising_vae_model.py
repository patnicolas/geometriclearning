__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

from typing import AnyStr
from dl.model.neural_model import NeuralModel
from dl.model.vae_model import VAEModel
import torch

import logging
logger = logging.getLogger('dl.model.DenoisingVAEModel')


def noise_func(x: torch.Tensor) -> torch.Tensor:
    noise_factor = 0.2
    x_with_noise = x + noise_factor * torch.randn_like(x)
    return torch.clamp(x_with_noise, min=0.0, max=1.0)


class DenoisingVAEModel(VAEModel):
    def __init__(self,
                 model_id: AnyStr,
                 encoder: NeuralModel,
                 latent_size: int) -> None:
        """
        Constructor for the variational neural network
        @param model_id: Identifier for this model
        @type model_id: str
        @param encoder: Neural network encoder
        @type encoder: NeuralModel
        @param latent_size: Size of the latent space
        @type latent_size: int
        """
        super(DenoisingVAEModel, self).__init__(model_id, encoder, latent_size, noise_func)