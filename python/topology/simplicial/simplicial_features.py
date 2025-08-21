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

# Standard Library imports
from typing import AnyStr, Dict, Tuple, List, Self
from dataclasses import dataclass
# 3rd Party import
import numpy as np
from toponetx.classes.simplicial_complex import SimplicialComplex


@dataclass
class SimplicialFeatures:
    all_features: Tuple[Dict[Tuple, np.array]]

    def node_features(self) -> Dict[Tuple, np.array]:
        return self.all_features[0]

    def edge_features(self) -> Dict[Tuple, np.array]:
        return self.all_features[1]

    def face_features(self) -> Dict[Tuple, np.array]:
        return self.all_features[2]

    def tetrahedron_features(self) -> Dict[Tuple, np.array]:
        return self.all_features[3]

    @classmethod
    def build(cls, simplicial_complex: SimplicialComplex, simplex_features: List[AnyStr]) -> Self:
        collected_features = [simplicial_complex.get_simplex_attributes(simplex_feature, rank=idx)
                              for idx, simplex_feature in enumerate(simplex_features)]
        return cls(tuple(collected_features))

    def show(self, num_entries: int) -> AnyStr:
        node_names = ['Node', 'Edge', 'Face', 'Tetrahedron']
        display = []

        for idx, features in enumerate(self.all_features):
            features = list(features.items())[0:num_entries]
            entry = ', '.join([f'{str(el[0])}: {str(el[1])}' for el in features])
            display.append(f'{node_names[idx]} Features:\n{entry}')
        return '\n'.join(display)
