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

import logging
import python
# Library imports
from play import Play
from topology.homology.persistence_barcodes import PersistentBarcodes
from topology.homology.shaped_data_generator import ShapedDataGenerator, ShapedDataDisplay
from topology.homology.persistence_diagrams import PersistenceDiagrams


class PersistentBarcodesPlay(Play):
    """
    Wrapper to implement the evaluation of persistence diagrams as defined in Substack article:
    "Understanding Data Through Persistence Diagrams"

    References:
    - Article: https://patricknicolas.substack.com/p/understanding-data-through-persistence
    - Implementation
      https://github.com/patnicolas/geometriclearning/blob/main/python/topology/homology/persistent_barcodes.py
      https://github.com/patnicolas/geometriclearning/blob/main/python/topology/homology/shaped_data_generator.py
    - Evaluation
      https://github.com/patnicolas/geometriclearning/blob/main/play/persistent_barcodes_play.py

    The features are implemented by the class PersistentBarcodes in the source file topology/persistent_barcodes.py
    The execution of the tests follows the same order as in the Substack article
    """
    def __init__(self, shaped_data_generator: ShapedDataGenerator) -> None:
        super(PersistentBarcodesPlay, self).__init__()
        self.shaped_data_display =  ShapedDataDisplay(shaped_data_generator)

    def play(self) -> None:
        self.__execute(noise_factor=0.02)
        self.__execute(noise_factor=0.45)

    """  ----------------  Private Helper Method --------------------- """

    def __execute(self, noise_factor: float) -> None:
        # Step 1 Generate shaped (Circle + Gaussian Noise) data
        _, shaped_data, shape_type = self.shaped_data_display.get_data(props={'n': 200}, noise=noise_factor, limit=20)

        # Step 2 Generate and display persistence diagram (component Birth-Death)
        persistence_diagrams = PersistenceDiagrams(data=shaped_data, data_shape=shape_type)
        persistence_diagrams.display()

        # Step 3 Generate bar codes
        barcodes = PersistentBarcodes(shaped_data)
        barcodes.display()


if __name__ == '__main__':
    try:
        barcodes_play = PersistentBarcodesPlay(ShapedDataGenerator.CIRCLE)
        barcodes_play.play()
    except (ValueError, TypeError) as e:
        logging.error(e)

