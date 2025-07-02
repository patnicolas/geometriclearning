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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from dataclasses import dataclass
from typing import AnyStr, NoReturn
import numpy as np
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.backend as gs  # Numpy
__all__ = ['SPDTestData', 'BinaryLRManifold']


@dataclass
class SPDTestData:
    """
    Wrapper for the test data for the Symmetric Positive Define matrices
    X- Features data
    y- Label
    """
    X: np.array
    y: np.array

    def shape(self) -> int:
        """
        Compute the shape of this SPD data set
        @return shape of test data
        @rtype int
        """
        return self.X.shape

    def n_samples(self) -> int:
        """
        Accessed the number of samples
        @return Number of samples
        @rtype int
        """
        return len(self.X)

    def flatten(self) -> NoReturn:
        """
        Flatten the 2 dimension feature into a single dimension feature
        """
        shape = self.X.shape
        assert(len(shape) == 3, f'Incorrect shape {len(shape)} for the SPD Training data')
        shape2= shape[1]*shape[2]
        self.X = self.X.reshape(shape[0], shape2)

    def load_indexed_data(self, features_path: AnyStr, label_path: AnyStr) -> NoReturn:
        train_data = np.genfromtxt(features_path, delimiter=',', skip_header=0)
        train_pairs = [(x[0], x[1:]) for x in train_data]
        labels = np.genfromtxt(label_path, delimiter=',', skip_header=0)
        label_map = [{y[0]: y[1]} for y in labels]
        labeled_data = []
        for index, features in train_pairs:
            label = label_map[index]
            labeled_data.append(features)

    def __str__(self) -> AnyStr:
        return f'X[0]: {self.X[0]}\ny[0]: {self.y[0]}'


class BinaryLRManifold(object):
    """
    Class that wraps the evaluation of Geomstats functions to evaluate the binary logistic regression
    on a manifold as the group of Symmetric Positive Definite (SPD) matrices and compare with the
    default logistic regression on the Euclidean space.

    The methods evaluate_euclidean and evaluate_spd apply the Scikit-learn cross validation to verify
    that the mean value of cross-validation score for randomly generated SPD matrices is close to 0.5
    """
    def __init__(self, n_features: int, n_samples: int):
        """
        Constructor for the Binary Logistic Regression classifier
        @param n_features: Number of features
        @type n_features: int
        @param n_samples: Number of samples
        @type n_samples: int
        """
        assert(n_features > 1, f'Number of features {n_features} should be > 1')
        assert (n_samples > 0, f'Number of features {n_samples} should be > 0')

        self.n_features = n_features
        self.n_samples = n_samples

    def generate_random_data(self) -> SPDTestData:
        """
        Generate a random set of features data and labels
        @return A SPD test data
        @rtype SPDTestData
        """
        y = np.stack([np.random.randint(0, 2) for _ in range(self.n_samples)])
        X = np.stack([self.__generate_sps_data() for _ in range(self.n_samples)])
        return SPDTestData(X, y)

    def create_spd(self, riemannian_metric: RiemannianMetric) -> SPDMatrices:
        """
        Create a pair of SPD matrices using one of the Riemann metric (either affine metric or
        logistic Euclidean metric
        @param riemannian_metric: Riemann metric used to generate the SPD matrix
        @type riemannian_metric: RiemannianMetric {SPDAffineMetric, SPDLogEuclideanMetric}
        @return A pair of manifold, SPD matrices
        @rtype Tuple of SPDMatrices
        """
        spd = SPDMatrices(self.n_features, equip=False)
        spd.equip_with_metric(riemannian_metric)
        return spd

    @staticmethod
    def evaluate_euclidean(spd_test_data: SPDTestData) -> Dict[AnyStr, np.array]:
        """
        Evaluate the logistic regression on Euclidean space
        @param spd_test_data: Test data for SPD metric
        @type spd_test_data: SPDTestData
        @return A dictionary of scoring values
        @rtype Dict
        """
        model = LogisticRegression()
        spd_test_data.flatten()
        return cross_validate(model, spd_test_data.X, spd_test_data.y)

    @staticmethod
    def evaluate_spd(spd_test_data: SPDTestData, spd_matrices: SPDMatrices) -> Dict[AnyStr, np.array]:
        """
        Evaluate the logistic regression on tangent space of a SPD manifold
        @param spd_test_data: Test data for SPD metric
        @type spd_test_data: SPDTestData
        @param spd_matrices: SPD matrices
        @type spd_matrices: SPDMatrices
        @return A dictionary of scoring values
        @rtype Dict
        """
        from geomstats.learning.preprocessing import ToTangentSpace
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            steps=[('features', ToTangentSpace(space = spd_matrices)),
                ('classifier', LogisticRegression())]
        )
        return cross_validate(pipeline, spd_test_data.X, spd_test_data.y)

    """ ---------------------------  Private helper methods -----------------------  """
    def __generate_sps_data(self) -> np.array:
        epsilon = 1e-6
        mat = np.random.rand(self.n_features, self.n_features)
        mat = (mat + mat.T)/2
        eigenvalues = np.linalg.eigvals(mat)
        min_eigen = np.min(eigenvalues)
        if min_eigen <= 0:
            mat += (np.eye(self.n_features)*(-min_eigen + epsilon))
        return mat




