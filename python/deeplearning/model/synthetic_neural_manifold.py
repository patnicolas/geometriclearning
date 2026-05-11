__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

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

# Standard Library imports
from typing import Tuple, AnyStr
# 3rd Party Libray imports
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.manifold import Isomap


class NeuralActivityGenerator(object):
    """
        Generator for synthetic Neural Activity using a S1 manifold latent space and
        a random Poisson distribution for continuous representation of neural spikes.

        The random Poisson distribution is defined as
        ..math:: Poisson(k; \lambda)=\frac{\lambda^k e^{-\lambda}}{k!} \ \ \ \ \lambda = \frac{1}{2}[(\tilde{x}-x_{L})^2 + (\tilde{y}-y_{L})^2]
        For events with an expected separation :math:`\lambda` the Poisson distribution :math:`Poisson(k; \lambda)`
        describes the probability of :math:`k` events occurring within the observed interval :math:`\lambda`.
    """
    def __init__(self,
                 n_neurons: int,
                 n_timesteps: int = 512,
                 firing_rate_factor: float = 20.0,
                 velocity: float = 0.05) -> None:
        """
        Constructor for the generator of synthetic neural activity
        @param n_neurons:  Number of neurons
        @type n_neurons: int
        @param n_timesteps: Number of time steps for sampling and interpolation
        @type n_timesteps: int
        @param firing_rate_factor: Scaling factor for the firing rate of Poisson distribution
        @type firing_rate_factor: float
        @param velocity: Define the duration of the sampling as T = timesteps * velocity
        @type velocity: float
        """
        self.n_neurons = n_neurons
        self.n_timesteps = n_timesteps
        self.firing_rate_factor = firing_rate_factor
        self.velocity = velocity

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dunda method that wraps the generation of the spikes trains (one per neuron) and the original latent
        path.
        @return: Tuple of spike trains and latent space
        @rtype: Tuple[np.ndarray, np.ndarray]
        """
        # 1. Generate the latent manifold trajectory S1 group
        t = np.linspace(0, self.n_timesteps * self.velocity, self.n_timesteps)
        latent_x = np.cos(t)
        latent_y = np.sin(t)
        latent_path = np.stack([latent_x, latent_y], axis=1)

        # 2. Assign each neuron a preferred location on the manifold S1
        angles = np.linspace(0, 2 * np.pi, self.n_neurons)
        pref_x = np.cos(angles)
        pref_y = np.sin(angles)

        # 3. Calculate firing rates based on distance to preferred location (Tuning Curves)
        # Rate = max_rate * exp(-distance^2 / 2*sigma^2)
        spikes = np.zeros((self.n_timesteps, self.n_neurons))
        for i in range(self.n_neurons):
            dist_sq = (latent_x - pref_x[i]) ** 2 + (latent_y - pref_y[i]) ** 2
            firing_rate = self.firing_rate_factor * np.exp(-dist_sq / 0.5)
            # Generate Poisson spikes
            spikes[:, i] = np.random.poisson(firing_rate * 0.02)

        return spikes, latent_path

    def __str__(self) -> AnyStr:
        """
        Short hand descriptor for display in plots
        @return: Basic, minimum configuration of the Spikes generator
        @rtype: str
        """
        return (f"Synthetic Neural Activity - {self.n_neurons} Neurons, {self.n_timesteps} "
                f"steps, Firing Rate Factor: {self.firing_rate_factor}")


class SyntheticNeuralManifold(object):
    """
    Neural Manifold using synthetic neural activity 
    THe spikes are smooths using a simple !D Gaussian distribution with the standard deviation as a parameter.
    The Manifold is visualized using an Isomap in 2 dimension for which the number of neighbors is also a parameters

    Important note: The Isometric feature mapping assumes the manifold is convex (positive curvature). This assumption
    makes sense in this case as the spikes are synthetically generated through the latent circle.
    """
    def __init__(self, neural_activity_generator: NeuralActivityGenerator) -> None:
        """
        Constructor for this synthetic neural manifold 
        @param neural_activity_generator: The generator of neural spikes train
        @type neural_activity_generator: NeuralActivityGenerator
        """
        self.neural_activity_generator = neural_activity_generator

    def __call__(self, sigma: float, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generation of the neural activity and the manifold using a Gaussian filter for converting the neural spikes
        into a continuous function and an Isomap manifold model
        
        @param sigma: Standard deviation for the Gaussian smoothing function
        @type sigma: float
        @param n_neighbors: Number of neighbors used in the Isomap
        @type n_neighbors: int
        @return: Spike trains (one for each neuron) 
        @rtype: Tuple[np.ndarray, np.ndarray]:
        """
        spikes, _ = self.neural_activity_generator()
        smoothed_spikes = gaussian_filter1d(spikes, sigma=sigma, axis=0)
        embedding = Isomap(n_components=2, n_neighbors=n_neighbors)
        return spikes, embedding.fit_transform(smoothed_spikes)

    def get_num_neurons(self) -> int:
        return self.neural_activity_generator.n_neurons

    def get_descriptor(self) -> AnyStr:
        return str(self.neural_activity_generator)

    @staticmethod
    def reconstruction_error(spikes: np.ndarray, sigma: float) -> None:
        """
        Method to evaluate the optimal number of neighbors for the Isomap algorithm
        @param spikes: Spikes trains
        @type spikes: np.ndarray
        @param sigma: Standard deviation for the Gaussian smoothing function
        @type sigma: float
        """
        smoothed_spikes = gaussian_filter1d(spikes, sigma=sigma, axis=0)
        errors = []
        for k in range(2, 32):
            embedding = Isomap(n_components=2, n_neighbors=k)
            embedding.fit(smoothed_spikes)
            errors.append(embedding.reconstruction_error())
        print(errors)
