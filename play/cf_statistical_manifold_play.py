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

# Python standard library imports
from typing import Tuple
import logging
# 3rd Party imports
import torch
from geomstats.information_geometry.base import InformationManifoldMixin
from geomstats.information_geometry.exponential import ExponentialDistributions
from geomstats.information_geometry.poisson import PoissonDistributions
from geomstats.information_geometry.geometric import GeometricDistributions
from geometry.information_geometry import InformationGeometryException
from geometry.information_geometry.cf_statistical_manifold import CFStatisticalManifold
# Library imports
from play import Play
import python


class CfStatisticalManifoldPlay(Play):
    """
    Source code related to the Substack article 'Geometry of Closed-Form Statistical Manifolds'.
    Reference: https://patricknicolas.substack.com/p/geometry-of-closed-form-statistical

    Statistical Manifolds:
    https://github.com/patnicolas/geometriclearning/blob/main/python/geometry/information_geometry/cf_statistical_manifold.py

    The features are implemented by the class CfStatisticalManifold in the source file
                  python/geometry/information_geometry/cf_statistical_manifold.py
    The class CfStatisticalManifoldPlay is a wrapper of the class CfStatisticalManifold
    The execution of the tests follows the same order as in the Substack article
    """
    def __init__(self, info_manifold: InformationManifoldMixin, bounds: Tuple[float, float]) -> None:
        super(CfStatisticalManifoldPlay, self).__init__()
        self.statistical_manifold = CFStatisticalManifold(info_manifold, bounds)

    def play(self) -> None:
        self.play_random_samples(n_samples=8)
        self.play_metric_matrix()
        self.play_exp_map()
        self.play_log_map()

    def play_random_samples(self, n_samples: int) -> None:
        """
        Source code related to Substack article 'Geometry of Closed-Form Statistical Manifolds' - Code snippet 4
        Ref: https://patricknicolas.substack.com/p/geometry-of-closed-form-statistical

        @param n_samples: Number of samples
        @type n_samples: int
        """
        samples = self.statistical_manifold.samples(n_samples=n_samples)

        assert self.statistical_manifold.belongs(list(samples))
        samples_str = '\n'.join([str(x) for x in samples])
        distribution_type = self.statistical_manifold.info_manifold.__class__.__name__
        logging.info(f'\n{distribution_type} Distribution Manifold 8 random samples \n{samples_str}')

    def play_metric_matrix(self) -> None:
        """
        Source code related to Substack article 'Geometry of Closed-Form Statistical Manifolds' - Code snippet 6
        Ref: https://patricknicolas.substack.com/p/geometry-of-closed-form-statistical
        """
        metric = self.statistical_manifold.metric_matrix()
        distribution_type = self.statistical_manifold.info_manifold.__class__.__name__
        logging.info(f'\n{distribution_type} Distribution Fisher metric: {metric}')

    def play_exp_map(self) -> None:
        """
        Source code related to Substack article 'Geometry of Closed-Form Statistical Manifolds' - Code snippet 8
        Ref: https://patricknicolas.substack.com/p/geometry-of-closed-form-statistical
        """
        tgt_vector = torch.Tensor([0.5])
        base_point = self.statistical_manifold.samples(1)
        end_point = self.statistical_manifold.exp(tgt_vector, base_point)
        distribution_type = self.statistical_manifold.info_manifold.__class__.__name__
        logging.info(f'\n{distribution_type} Distribution Manifold End point: {end_point}')

    def play_log_map(self) -> None:
        """
        Source code related to Substack article 'Geometry of Closed-Form Statistical Manifolds' - Code snippet 10
        Ref: https://patricknicolas.substack.com/p/geometry-of-closed-form-statistical
        """
        random_points = self.statistical_manifold.samples(2)
        base_point = random_points[0]
        manifold_point = random_points[1]
        tgt_vector = self.statistical_manifold.log(manifold_point, base_point)
        distribution_type = self.statistical_manifold.info_manifold.__class__.__name__
        logging.info(f'\n{distribution_type} Distribution Manifold Tangent Vector\nBase:{base_point} '
                     f'to:{manifold_point}: {tgt_vector}')


if __name__ == '__main__':
    try:
        # Test 1 - Evaluation code Substack article 'Geometry of Closed-Form Statistical Manifolds' - Code snippet 4
        cf_statistical_manifold_play = CfStatisticalManifoldPlay(ExponentialDistributions(equip=True), bounds=(-2, 2))
        cf_statistical_manifold_play.play_random_samples(n_samples=8)

        cf_statistical_manifold_play = CfStatisticalManifoldPlay(GeometricDistributions(equip=True), bounds=(1, 10))
        cf_statistical_manifold_play.play_random_samples(n_samples=4)

        # Test 2 - Evaluation code Substack article 'Geometry of Closed-Form Statistical Manifolds' - Code snippet 6
        cf_statistical_manifold_play = CfStatisticalManifoldPlay(ExponentialDistributions(equip=True), bounds=(0, 2))
        cf_statistical_manifold_play.play_metric_matrix()

        cf_statistical_manifold_play = CfStatisticalManifoldPlay(PoissonDistributions(equip=True), bounds=(0, 20))
        cf_statistical_manifold_play.play_metric_matrix()

        # Test 3 - Evaluation code Substack article 'Geometry of Closed-Form Statistical Manifolds' - Code snippet 8
        cf_statistical_manifold_play = CfStatisticalManifoldPlay(ExponentialDistributions(equip=True), bounds=(0, 2))
        cf_statistical_manifold_play.play_exp_map()

        cf_statistical_manifold_play = CfStatisticalManifoldPlay(GeometricDistributions(equip=True), bounds=(1, 10))
        cf_statistical_manifold_play.play_exp_map()

        # Test 4 - Evaluation code Substack article 'Geometry of Closed-Form Statistical Manifolds' - Code snippet 8
        cf_statistical_manifold_play = CfStatisticalManifoldPlay(ExponentialDistributions(equip=True), bounds=(0, 2))
        cf_statistical_manifold_play.play_log_map()

        cf_statistical_manifold_play = CfStatisticalManifoldPlay(GeometricDistributions(equip=True), bounds=(1, 10))
        cf_statistical_manifold_play.play_log_map()
    except AttributeError as e:
        logging.error(e)
        assert False
    except InformationGeometryException as e:
        logging.error(e)
        assert False

