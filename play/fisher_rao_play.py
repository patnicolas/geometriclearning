__author__ = "Patrick R. Nicolas"
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


# Python standard library imports
from typing import Tuple
import logging
# 3rd party library import
import torch
from geomstats.information_geometry.base import InformationManifoldMixin
from geomstats.information_geometry.exponential import ExponentialDistributions
from geomstats.information_geometry.poisson import PoissonDistributions
from geomstats.information_geometry.geometric import GeometricDistributions
from geomstats.information_geometry.binomial import BinomialDistributions
# Library imports
from play import Play
from geometry.information_geometry.fisher_rao import FisherRao
import python


class FisherRaoPlay(Play):
    """
    Source code related to the Substack article 'Shape Your Models with the Fisher-Rao Metric'.

    Article: https://patricknicolas.substack.com/p/shape-your-models-with-the-fisher
    Fisher-Rao:
        https://github.com/patnicolas/geometriclearning/blob/main/python/geometry/information_geometry/fisher_rao.py

    The features are implemented by the class FisherRao in the source file
                  python/geometry/information_geometry/fisher_rao.py
    The class FisherRaoPlay is a wrapper of the class FisherRao
    The execution of the tests follows the same order as in the Substack article
    """
    def __init__(self, info_manifold: InformationManifoldMixin, bounds: Tuple[float, float]) -> None:
        super(FisherRaoPlay, self).__init__()
        self.fisher_rao = FisherRao(info_manifold, bounds)

    def play(self) -> None:
        self.play_sample_distribution(n_samples=8)
        self.play_distance()
        _v = torch.Tensor([1.0])
        _w = torch.Tensor([-1.0])
        self.play_inner_product(_v, _w)

    def play_sample_distribution(self, n_samples: int) -> None:
        """
        Source code related to Substack article 'Shape Your Models with the Fisher-Rao Metric' - Code snippet 4
        Ref: https://patricknicolas.substack.com/p/shape-your-models-with-the-fisher

        @param n_samples: Number of sampled parameters
        @type n_samples: int
        """
        random_samples = self.fisher_rao.samples(n_samples=n_samples)
        assert self.fisher_rao.belongs(list(random_samples))

        results = ', '.join([f'{(float(x)):.4f}' for x in random_samples])
        stats_manifold_name = self.fisher_rao.info_manifold.__class__.__name__
        logging.info(f"{stats_manifold_name} samples:\n{results}")

    def play_distance(self) -> None:
        """
        Source code related to Substack article 'Shape Your Models with the Fisher-Rao Metric' - Code snippet 4
        Ref: https://patricknicolas.substack.com/p/shape-your-models-with-the-fisher
        """
        values = self.fisher_rao.samples(2)
        distance = self.fisher_rao.distance(values[0], values[1])
        logging.info(f'd({float(values[0]):.4f}, {float(values[1]):.4f}) = {float(distance):.4f}')
        self.fisher_rao.visualize_diff(parameter1=values[0], parameter2=values[1], param_label=r"$\theta$")

    def play_inner_product(self, v: torch.Tensor, w: torch.Tensor) -> None:
        """
        Source code related to Substack article 'Shape Your Models with the Fisher-Rao Metric' - Code snippet 6
        Ref: https://patricknicolas.substack.com/p/shape-your-models-with-the-fisher
        """
        point = [torch.Tensor(x) for x in self.fisher_rao.samples(2)]
        inner_product = self.fisher_rao.inner_product(point[0], v, w)
        stats_manifold_name = self.fisher_rao.info_manifold.__class__.__name__
        logging.info(f'{stats_manifold_name} {float(v)} dot {float(w)} = {float(inner_product):.3f}')


if __name__ == '__main__':
    fisher_rao_plays = [
        FisherRaoPlay(info_manifold=ExponentialDistributions(equip=True), bounds=(-2.0, 2.0)),
        FisherRaoPlay(info_manifold=GeometricDistributions(equip=True), bounds=(1.0, 12.0)),
        FisherRaoPlay(info_manifold=PoissonDistributions(equip=True), bounds=(0.0, 20.0)),
        FisherRaoPlay(info_manifold=BinomialDistributions(equip=True, n_draws=8), bounds=(0.0, 20.0))
    ]

    # Test 1 for 'Shape Your Models with the Fisher-Rao Metric' - Code snippet 3
    for fisher_rao_play in fisher_rao_plays:
        fisher_rao_play.play_sample_distribution(n_samples=8)

    # Test 2 for 'Shape Your Models with the Fisher-Rao Metric' - Code snippet 4
    for fisher_rao_play in fisher_rao_plays:
        fisher_rao_play.play_distance()

    # Test 3 for 'Shape Your Models with the Fisher-Rao Metric' - Code snippet 6
    v = torch.Tensor([1.0])
    w = torch.Tensor([-1.0])
    for fisher_rao_play in fisher_rao_plays:
        fisher_rao_play.play_inner_product(v, w)