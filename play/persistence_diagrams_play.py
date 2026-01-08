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
import logging
from typing import Dict, Any, AnyStr
# Library imports
from play import Play
from topology.homology.shaped_data_generator import ShapedDataGenerator, ShapedDataDisplay
from topology.homology.persistence_diagrams import PersistenceDiagrams
import python


class PersistenceDiagramPlay(Play):
    """
    Wrapper to implement the evaluation of persistence diagrams as defined in Substack article:
    "Understanding Data Through Persistence Diagrams"

    References:
    - Article:
    - Implementation
      https://github.com/patnicolas/geometriclearning/blob/main/python/topology/homology/persistence_diagrams.py
      https://github.com/patnicolas/geometriclearning/blob/main/python/topology/homology/shaped_data_generator.py
    - Evaluation
      https://github.com/patnicolas/geometriclearning/blob/main/play/persistence_diagrams_play.py

    The features are implemented by the class PersistenceDiagrams in the source file
                       python/topology/persistence_diagrams.py
    The class persistence_diagramsPlay is a wrapper of the class persistence_diagrams
    The execution of the tests follows the same order as in the Substack article
    """
    def __init__(self, props: Dict[AnyStr, Any],  shaped_data_generator: ShapedDataGenerator) -> None:
        super(PersistenceDiagramPlay, self).__init__()
        self.persistence_diagrams =  PersistenceDiagrams.build(props=props,
                                                               shaped_data_generator=shaped_data_generator)

    def play(self) -> None:
        self.persistence_diagrams.display()

    @staticmethod
    def play_shaped_data_generator() -> None:
        uniform_data_display = ShapedDataDisplay(ShapedDataGenerator.UNIFORM)
        uniform_data_display.__call__({'n': 200})

        sphere_data_display = ShapedDataDisplay(ShapedDataGenerator.SPHERE)
        sphere_data_display.__call__({'n': 200}, noise=0.25)

        swiss_roll_data_display = ShapedDataDisplay(ShapedDataGenerator.SWISS_ROLL)
        swiss_roll_data_display.__call__({'n': 200}, noise=0.25)

        torus_data_display = ShapedDataDisplay(ShapedDataGenerator.TORUS)
        torus_data_display.__call__({'n': 200}, noise=0.25)


if __name__ == '__main__':
    try:
        """
        Sequence of tests used in the substack article "Understanding Data Through Persistence Diagrams"
        """
        # 3D plot of noisy randomly uniform data or shaped from a sphere, swiss roll or torus.
        PersistenceDiagramPlay.play_shaped_data_generator()

        # Persistence diagrams for Uniform randomly distributed data
        persistence_diagram_play = PersistenceDiagramPlay(props={'n': 256},
                                                          shaped_data_generator=ShapedDataGenerator.UNIFORM)
        persistence_diagram_play.play()

        # Persistence diagrams for sphere-shaped data set
        persistence_diagram_play = PersistenceDiagramPlay(props={'n': 512, 'noise': 0.0},
                                                          shaped_data_generator = ShapedDataGenerator.SPHERE)
        persistence_diagram_play.play()

        # Persistence diagrams for swiss roll-shaped data set
        persistence_diagram_play = PersistenceDiagramPlay(props={'n': 384, 'noise': 0.2},
                                                          shaped_data_generator=ShapedDataGenerator.SWISS_ROLL)
        persistence_diagram_play.play()

        # Persistence diagrams for torus-shaped data set
        persistence_diagram_play = PersistenceDiagramPlay(props={'n': 256, 'c': 20, 'a': 15, 'noise': 0.65},
                                                          shaped_data_generator=ShapedDataGenerator.TORUS)
        persistence_diagram_play.play()
    except (ValueError, TypeError) as e:
        logging.error(e)




