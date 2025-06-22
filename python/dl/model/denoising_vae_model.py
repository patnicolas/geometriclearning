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

from typing import AnyStr, Callable
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
                 latent_dim: int,
                 noise_func: Callable[[torch.Tensor], torch.Tensor] = None
                 ) -> None:
        """
        Constructor for the variational neural network
        @param model_id: Identifier for this model
        @type model_id: str
        @param encoder: Neural network encoder
        @type encoder: NeuralModel
        @param latent_dim: Size of the latent space
        @type latent_dim: int
        @param noise_func: Optional function to add noise to input data (features)
        @param noise_func: Callable (noise_factor, input)
        """
        self.noise_func = noise_func
        super(DenoisingVAEModel, self).__init__(model_id, encoder, latent_dim)