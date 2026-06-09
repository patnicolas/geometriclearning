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

from typing import List
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import python
from topology.homology.shaped_data_generator import ShapedDataDisplay, ShapedDataGenerator


class SyntheticFiltration(object):
    """
    Class to illustrate Filtration process using points distributed over S1 group (circle) randomly
    using a normal distribution.
    The synthetic filtration has the following parameters:
    - Number of data points - num_data_points
    - Variance of the normal distribution to locate the points - s1_variance
    - Number of stages in the filtration - num_stages
    - Rate of increase of the diameter of the ball - epsilon_delta
    """
    def __init__(self, num_data_points: int, noise_factor: float, num_stages: int, epsilon_delta: float = 0.2) -> None:
        if num_data_points < 2 or num_data_points > 64:
            raise ValueError("num_data_points should be between 2 and 64")
        if noise_factor < 0.0 or noise_factor > 1.0:
            raise ValueError("s1_variance should be between 0.0 and 1")
        if num_stages < 4 or num_stages > 12:
            raise ValueError("num_stages should be between 4 and 12")
        if epsilon_delta < 0.1 or epsilon_delta > 2.6:
            raise ValueError("epsilon_delta should be between 0.2 and 2.6")

        self.num_data_points = num_data_points
        self.noise_factor = noise_factor
        self.num_stages = num_stages
        self.epsilon_delta = epsilon_delta

    def __call__(self, data: np.ndarray) -> None:
        epsilons = self.__generate_epsilons()
        self.__show(data, epsilons)

    """ --------------------------  Private helper methods -------------------  """

    def __generate_epsilons(self) -> List[float]:
        return [self.epsilon_delta * (1 + n) for n in range(self.num_stages)]

    def __show(self, data: np.ndarray, epsilons: List[float]):
        dist_matrix = squareform(pdist(data))
        fig, axes = plt.subplots(2, 4, figsize=(14, 12))
        axes = axes.flatten()
        titles = [rf"$\epsilon = {epsilon:.1f}$" for epsilon in epsilons]

        for idx, eps in enumerate(epsilons):
            title = titles[idx]
            # Plot the growing balls around each point
            for i in range(len(data)):
                circle = plt.Circle((data[i, 0],
                                     data[i, 1]),
                                    eps,
                                    color='gainsboro',
                                    alpha=0.6,
                                    ec='gray',
                                    linestyle='--')
                axes[idx].add_patch(circle)

            # Plot edges between points if their distance is less than 2 * eps
            for i in range(len(data)):
                for j in range(i + 1, len(data)):
                    if dist_matrix[i, j] <= 2 * eps:
                        axes[idx].plot([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]], color='blue', lw=2, zorder=2)

            # Plot the original 6 data points
            axes[idx].scatter(data[:, 0], data[:, 1], color='red', s=75, zorder=3)

            # Formatting
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].set_xlim(-2.5, 2.5)
            axes[idx].set_ylim(-2.5, 2.5)
            axes[idx].set_aspect('equal')
            axes[idx].grid(True, linestyle=':', alpha=0.6)

        plt.suptitle(f"{self.num_stages} stages Vietoris-Rips filtration over {self.num_data_points} "
                     f"points & noise factor {self.noise_factor}",
                     fontsize=17,
                     fontweight='bold',
                     y=0.93)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    try:
        noise_factor = 0.02
        sphere_data_display = ShapedDataDisplay(ShapedDataGenerator.CIRCLE)
        _, shaped_data, _ = sphere_data_display.get_data(props={'n': 200}, noise=noise_factor, limit=20)
        sphere_data_display(props={'n': 200}, noise=noise_factor)
        synthetic_filtration = SyntheticFiltration(num_data_points=10,
                                                   noise_factor=noise_factor,
                                                   num_stages=8,
                                                   epsilon_delta=0.1)
        synthetic_filtration(shaped_data)
    except ValueError as e:
        logging.error(e)


"""
# Compute pairwise distances to find exact birth/death thresholds
num_points = 10
num_stages = 6
data = generate_S1_points(num_points=num_points, variance=0.8)
dist_matrix = squareform(pdist(data))

# 2. Define 3 key filtration values (epsilons) to visualize
# eps_1: Components are isolated (H0 = 6, H1 = 0)
# eps_2: Loop forms (H0 = 1, H1 = 1)
# eps_3: Loop fills in (H0 = 1, H1 = 0)
epsilons = [0.2*(1 + n) for n in range(num_stages)]


titles = [rf"$\epsilon = {epsilon}$" for epsilon in epsilons]
titles1 = [
    r"$\epsilon = 0.2$ (6 Components)",
    r"$\epsilon = 0.4$ (6 Components)",
    r"$\epsilon = 0.6$ (Loop Forms)",
    r"$\epsilon = 0.8$ (Loop Forms)",
    r"$\epsilon = 1.0$ (Loop Fills In)",
    r"$\epsilon = 1.2$ (Loop Fills In)"
]

fig, axes = plt.subplots(2, 3, figsize=(12, 12))
axes = axes.flatten()

for idx, eps in enumerate(epsilons):
    ax = axes[idx]
    title = titles[idx]
    # Plot the growing balls around each point
    for i in range(len(data)):
        circle = plt.Circle((data[i, 0], data[i, 1]), eps, color='gainsboro', alpha=0.6, ec='gray', linestyle='--')
        ax.add_patch(circle)

    # Plot edges between points if their distance is less than 2 * eps
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if dist_matrix[i, j] <= 2 * eps:
                ax.plot([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]], color='blue', lw=2, zorder=2)

    # Plot the original 6 data points
    ax.scatter(data[:, 0], data[:, 1], color='red', s=75, zorder=3)

    # Formatting
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)

plt.suptitle("Vietoris-Rips Filtration over 6 Points", fontsize=17, fontweight='bold', y=0.94)
plt.tight_layout()
plt.show()
"""


