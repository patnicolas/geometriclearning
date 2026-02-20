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

# Python standard library imports
from enum import StrEnum, unique
from typing import AnyStr
from abc import ABC, abstractmethod
import logging
import python
# 3rd party library imports
import umap
import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.manifold import TSNE
# Library import
from play import Play


@unique
class DataSrc(StrEnum):
    """
    Enumerator for the Source of data set
    """
    MNIST = 'mnist'
    IRIS = 'iris'


class NDLREval(ABC):
    """
    Base class to evaluate Nonlinear Dimensionality Reduction algorithm
    """
    def __init__(self, dataset_src: DataSrc) -> None:
        try:
            match dataset_src:
                case DataSrc.MNIST:
                    digits = load_digits()
                    self.data = digits.data
                    self.color = digits.target.astype(int)
                case DataSrc.IRIS:
                    images = load_iris()
                    self.data = images.data
                    self.names = images.target_names
                    self.color = images.target.astype(int)
        except Exception as e:
            raise Exception(f'Failed to load {str(dataset_src)} with {str(e)}')
        self.dataset_src = dataset_src

    @abstractmethod
    def __call__(self, cmap: AnyStr):
        pass


class TSneEval(NDLREval):
    """
    Class that wraps the evaluation of the t-SNE (t-distributed Stochastic Neighbor Embedding)
    Versions: Python  3.11,  SciKit-learn 1.5.1,  Matplotlib 3.9.1
    The dunda method __call__ is used to visualize the umap generated clusters in 2 dimension
    """
    def __init__(self, dataset_src: DataSrc, n_components: int) -> None:
        """
        Constructor for the evaluation of the t-distributed Stochastic Neighbor Embedding. An assert error is
        thrown if the number of components is not 2 or 3
        @param dataset_src: Source for the dataset (MNIST, IRIS,...)
        @type dataset_src: DataSrc
        @param n_components: Dimension of the tSNE analysis
        @type n_components: int
        """
        assert (n_components > 3 or n_components < 2, f'Number of components {n_components} is out of range')
        # Instantiate the selected data set
        super(TSneEval, self).__init__(dataset_src)
        # Instantiate the Sk-learn tSNE model
        self.t_sne = TSNE(n_components=n_components)

    def __call__(self, cmap: AnyStr) -> bool:
        """
        Visualize the tSNE data cluster in 2 or 3 dimension given a cmap
        @param cmap: Color theme or map identifier used in display
        @type cmap: str
        """
        import matplotlib.pyplot as plt

        try:
            embedded = self.t_sne.fit_transform(self.data)
            fig = plt.figure()
            if self.t_sne.n_components == 2:
                plt.scatter(embedded[:, 0], embedded[:, 1], c=self.color, cmap=cmap)
                plt.colorbar()
            else:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(*zip(*embedded[:, :2]), c=self.color, cmap=cmap)
            plt.title(f'tSNE {str(self.dataset_src)} {self.t_sne.n_components} components')
            plt.show()
            return True
        except Exception as e:
            logging.error(f't-SNE failed to visualize for {self.dataset_src}: {str(e)}')
            return False


class UMAPEval(NDLREval):
    """
    Class that wraps the evaluation of the Uniform Manifold Approximation and Projection (UMAP) algorithm.
    It relies on umap-learn Python module
    The dunda method __call__ is used to visualize the umap generated clusters in 2 dimension
    """
    def __init__(self, dataset_src: DataSrc, n_neighbors: int, min_dist: float) -> None:
        """
        Constructor for the evaluation of the Uniform Manifold Approximation and Projection
        algorithm.
        @param dataset_src: Source of the data set {i.e. IRIS)
        @type dataset_src: DataSrc
        @param n_neighbors: Number of neighbors associated with each alss
        @type n_neighbors: int
        @param min_dist: Minimum distance for UMAP
        @type min_dist: float
        """
        assert(2 <= n_neighbors <= 128, f'Number of neighbors {n_neighbors} is out of range [2, 128]')
        assert(1e-5 < min_dist < 0.2, f'Minimum distance {min_dist} is out of range')

        # Instantiate the selected data set
        super(UMAPEval, self).__init__(dataset_src)
        # Instantiate the UMAP model
        self.umap = umap.UMAP(random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)

    def __call__(self, cmap: AnyStr) -> bool:
        """
        Visualize the UMAP data cluster given a cmap, minimum distance and estimated number of neighbors
        @param cmap: Color theme or map identifier used in display
        @type cmap: str
        """
        import matplotlib.pyplot as plt
        try:
            embedding = self.umap.fit_transform(self.data)
            x = embedding[:, 0]
            y = embedding[:, 1]
            n_ticks = 10
            plt.scatter(x=x, y=y, c=self.color, cmap=cmap, s=4.0)
            plt.colorbar(boundaries=np.arange(n_ticks+1) - 0.5).set_ticks(np.arange(n_ticks))
            plt.title(f'UMAP {self.dataset_src} {self.umap.n_neighbors} neighbors, min_dist: {self.umap.min_dist}')
            plt.show()
            return True
        except Exception as e:
            logging.error(f'UMAP failed to visualize for {self.dataset_src}:  {str(e)}')
            return False


class UMAPPlay(Play):
    def __init__(self, tnse_eval: TSneEval, umap_eval: UMAPEval) -> None:
        super(UMAPPlay, self).__init__()
        self.tnse_eval = tnse_eval
        self.umap_eval = umap_eval

    def play_tsne(self) -> None:
        tsne_eval = TSneEval(dataset_src=DataSrc.MNIST, n_components=3)
        tsne_eval(cmap='Spectral')

        tsne_eval = TSneEval(dataset_src=DataSrc.IRIS, n_components=3)
        tsne_eval('Spectral')

    def play_umap(self) -> None:
        n_neighbors = 4
        min_dist = 0.8
        umap_eval = UMAPEval(dataset_src=DataSrc.MNIST, n_neighbors=n_neighbors, min_dist=min_dist)
        umap_eval(cmap='Spectral')

        n_neighbors = 40
        min_dist = 0.001
        umap_eval = UMAPEval(dataset_src=DataSrc.IRIS, n_neighbors=n_neighbors, min_dist=min_dist)
        umap_eval(cmap='Set1')



