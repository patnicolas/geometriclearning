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

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.mpl_axes import Axes
from ripser import ripser

from topology.homology.shaped_data_generator import ShapedDataDisplay, ShapedDataGenerator


class PersistentBarcodes(object):
    """ Labels and color for each Homology class """
    homology_labels = (('H0', 'blue'), ('H1', 'red'), ('H2', 'black'))

    def __init__(self, data: np.ndarray) -> None:
        # Ripser (Vietoris-Rips complex) library to process the data
        rips_result = ripser(data, maxdim=1)

        # Extract the diagrams using ripser dictionary
        self.diagrams = rips_result['dgms']
        # Create a visual representation if infinite filtration
        all_finite_deaths = [pt[1] for dgm in self.diagrams for pt in dgm if pt[1] != np.inf]
        self.infinity_val = 1.3 * max(all_finite_deaths) if all_finite_deaths else 1.0

    def display(self) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))

        barcode_cursor_y = 0
        # Record the tuple (tick Y-location, label) for each bar code
        barcode_entry: List[Tuple[float, str]] = []

        # Iterate through dimensions (H0 and H1)
        for dim, dgm in enumerate(self.diagrams):
            # Does not show the barcode for a given homology class if it is empty
            if len(dgm) > 0:
                H_start_y = barcode_cursor_y
                barcode_cursor_y = self.__barcode_per_homology_class(dgm, barcode_cursor_y, dim, ax)
                # Record the
                barcode_entry.append((H_start_y + (len(dgm) - 1) / 2, PersistentBarcodes.homology_labels[dim][0]))

                # Add a subtle visual divider between H0 and H1
                if dim < len(self.diagrams) - 1:
                    ax.axhline(barcode_cursor_y - 0.5, color='black', linestyle=':', alpha=0.5)

        # Styling the plot
        ticks, labels = zip(*barcode_entry)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=14, fontweight='bold')
        ax.set_xlabel("Filtration Value ($\epsilon$)", fontsize=12)
        ax.set_title(f"Persistence Barcode (10-Point Filtration) noise: 45%", fontsize=14, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Set x-limits safely
        ax.set_xlim(-0.05, self.infinity_val + 0.1)
        plt.tight_layout()
        plt.show()

    def __barcode_per_homology_class(self,
                                     dgm: np.ndarray,
                                     cur_barcode_y: int,
                                     dim: int,
                                     ax: Axes) -> int:
        # Sort features by lifetime (longest lasting first)
        sorted_dgm = sorted(dgm, key=lambda pt: pt[1] - pt[0], reverse=True)

        # Iterate across each barcode entry  (original points)
        for birth, death in sorted_dgm:
            # Compute the end of the bar taking care of the case of infinite
            plot_death = self.infinity_val if death == np.inf else death
            # Draw the horizontal bar
            ax.hlines(cur_barcode_y, birth, plot_death, colors=PersistentBarcodes.homology_labels[dim][1], linewidths=12)
            # If it goes to infinity, extends the line with dots
            if death == np.inf:
                ax.annotate('...',
                            xy=(plot_death, cur_barcode_y),
                            xytext=(plot_death + 0.02, cur_barcode_y),
                            fontsize=20)
            cur_barcode_y += 1
        return cur_barcode_y


if __name__ == '__main__':
    noise_factor = 0.45
    sphere_data_display = ShapedDataDisplay(ShapedDataGenerator.CIRCLE)
    _, shaped_data, _ = sphere_data_display.get_data(props={'n': 200}, noise=noise_factor, limit=20)
    barcodes = PersistentBarcodes(shaped_data)
    barcodes.display()
