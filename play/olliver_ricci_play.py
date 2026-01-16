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

from typing import List, Tuple, Callable
import logging
import python

import torch
# Library import
from play import Play
from geometry.discrete.olliver_ricci import OlliverRicci
from geometry.discrete import WassersteinException


class OlliverRicciPlay(Play):

    def __init__(self,
                 edge_index: List[Tuple[int, int]],
                 edge_weights: torch.Tensor | Callable[[int], float],
                 epsilon: float,
                 rc: Tuple[torch.Tensor, torch.Tensor] = None):
        super(OlliverRicciPlay, self).__init__()

        self.olliver_ricci = OlliverRicci(edge_index, edge_weights, epsilon, rc) \
            if isinstance(edge_weights, torch.Tensor) \
            else OlliverRicci.build(edge_index, edge_weights, epsilon, rc)

    def play(self):
        try:
            curvature = self.olliver_ricci.curvature(n_iters=100, early_stop_threshold=0.01)
            logging.info(f'\nCurvature:\n{curvature}')
        except (ValueError, WassersteinException) as e:
            logging.error(e)

def sphere_geodesics(n_edges: int) -> torch.Tensor:
    import math

    weights = []
    delta = math.pi / (6 * n_edges)
    geo_x = 3*math.pi/16
    geo_y = math.pi/6
    for i in range(n_edges):
        lat1, lon1 = map(math.radians, [geo_x, geo_y])
        lat2, lon2 = map(math.radians, [geo_x - delta, geo_y + delta])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(0.5*dlat) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(0.5*dlon) ** 2
        weights.append(2.0 * math.asin(math.sqrt(a)))
        if i % 2 == 0:
            geo_y -= delta
        else:
            geo_x -= delta
            geo_y += delta
    return torch.Tensor(weights)


if __name__ == '__main__':
    edge_index = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    edge_weights = torch.Tensor([0.8, 1.5, 2.6, 4.8, 2.2, 2.5, 6.1, 0.1, 3.8, 3.5])

    olliver_ricci_play = OlliverRicciPlay(edge_index, edge_weights, epsilon=0.02, rc=None)
    olliver_ricci_play.play()

    r = torch.tensor([0.1, 0.1, 0.2, 0.4], dtype=torch.float32)
    c = torch.tensor([0.1, 0.3, 0.4, 0.2], dtype=torch.float32)
    olliver_ricci_play = OlliverRicciPlay(edge_index, sphere_geodesics, epsilon=0.05, rc=(r, c))
    olliver_ricci_play.play()

