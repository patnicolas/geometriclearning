__author__ = "Patrick Nicolas"
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

import numpy as np
import matplotlib.pyplot as plt
from deeplearning.model.synthetic_neural_manifold import SyntheticNeuralManifold, NeuralActivityGenerator


class SyntheticNeuralManifoldPlay(object):
    def __init__(self, synthetic_neural_manifold: SyntheticNeuralManifold) -> None:
        self.synthetic_neural_manifold = synthetic_neural_manifold

    def show_spike_trains(self, spikes: np.ndarray) -> None:
        plt.figure(figsize=(10, 4))
        plt.imshow(spikes.T, aspect='auto', cmap='hot', interpolation='nearest')
        for i in range(self.synthetic_neural_manifold.get_num_neurons() + 1):
            plt.axhline(y=i - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.title(self.synthetic_neural_manifold.get_descriptor())
        plt.xlabel("Time")
        plt.ylabel("Neuron")
        plt.show()

    @staticmethod
    def show_latent_path(manifold_path: np.ndarray, sigma: float, n_neighbors: int) -> None:
        plt.style.use(['ggplot', 'dark_background'])

        fig = plt.figure(figsize=(8, 8))
        fig.set_facecolor('#2d3557')
        plt.scatter(manifold_path[:, 0], manifold_path[:, 1],
                    c=np.arange(len(manifold_path)), cmap='hsv', s=55, alpha=0.8)
        plt.title(f"Latent Neural Manifold - Gaussian sigma={sigma}, Isomap {n_neighbors} neighbors")
        plt.xlabel("Manifold Dim 1")
        plt.ylabel("Manifold Dim 2")
        plt.colorbar(label="Time Scale")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        # plt.savefig('manifold_projection.png')

    def play(self) -> None:
        sigma = 1.6
        n_neighbors = 32
        spikes, manifold_path = self.synthetic_neural_manifold(sigma=sigma, n_neighbors=n_neighbors)
        SyntheticNeuralManifoldPlay.show_latent_path(manifold_path, sigma=sigma, n_neighbors=n_neighbors)
        self.show_spike_trains(spikes)


if __name__ == '__main__':
    neural_activity_generator = NeuralActivityGenerator(n_neurons=512, n_timesteps=100, firing_rate_factor=96, velocity=0.05)
    synthetic_neural_manifold = SyntheticNeuralManifold(neural_activity_generator)
    play = SyntheticNeuralManifoldPlay(synthetic_neural_manifold)
    play.play()


