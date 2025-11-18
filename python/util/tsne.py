__author__ = "Patrick Nicolas"
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

import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
__all__ = ['TSNE']


class TSNE(object):
    """
        Apply the Stochastic Neigbhors Embedding to a two or 3 num_tfidf_features wordembedding.
        The main method, forward generates the appropriate image stored in 'images' directory
        and return the embedding
    """

    def __init__(self, n_components: int, cmap: str, fig_save_dir: str, title: str):
        """
        Constructor for T-SNE algorithm

        @param n_components:  Number of principal components (num_tfidf_features) of the embedding
        @type n_components: int
        @param cmap: Color map used for display
        @type cmap: str
        @param fig_save_dir: Directory the plot is stored
        @type fig_save_dir: str
        @param title: Title of the plot
        @type title: str
        """
        if n_components < 2 or n_components > 3:
            raise ValueError(f'TSNE: num of components {n_components} should be [2, 3]')

        self.t_sne = TSNE(n_components = n_components)
        self.cmap = cmap
        self.fig_save_dir = fig_save_dir
        self.title = title

    def forward(self, x: torch.Tensor) -> np.array:
        """
            Method to generate TSNE embedding with plot stored in a given directory
            @param x Input torch input_tensor
            @type x Torch Tensor
        """

        # Apply the TSNE transform
        embedded = self.t_sne.fit_transform(x.detach().numpy())
        n_points = len(embedded)
        # Just random colors
        colors = np.random.randn(n_points)
        # Set up the display - plotting
        fig = plt.figure()
        if self.t_sne.n_components == 2:
            plt.scatter(embedded[:,0], embedded[:,1], c= colors, cmap= self.cmap)
            plt.colorbar()
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*zip(*embedded[:,:2]), c=colors, cmap=self.cmap)
        plt.show()
        plt.title(self.title)
        # Save the plot
        if self.fig_save_dir:
            fig.savefig(self.fig_save_dir)
        return embedded


