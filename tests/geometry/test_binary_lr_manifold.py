import unittest

from geomstats.geometry.spd_matrices import SPDAffineMetric, SPDLogEuclideanMetric
from geometry.binary_lr_manifold import BinaryLRManifold
import numpy as np
import logging
import util

class BinaryLRManifoldTest(unittest.TestCase):

    def test_generate_data(self):
        n_samples = 2000
        n_features = 4
        binary_lr_on_spd = BinaryLRManifold(n_features, n_samples)
        train_data = binary_lr_on_spd.generate_random_data()
        shape = train_data.shape()
        logging.info(f'First 2 data points: {train_data.X[:2]}')
        logging.info(f'First 2 labels: {train_data.y[:2]}')


    def test_eval_spd_logistic_euclidean(self):
        n_samples = 6000
        n_features = 16
        binary_lr_on_spd = BinaryLRManifold(n_features, n_samples)
        train_data = binary_lr_on_spd.generate_random_data()
        logging.info(f'Training data shape: {train_data.shape()}')
        spd = binary_lr_on_spd.create_spd(SPDLogEuclideanMetric)
        result_dict = binary_lr_on_spd.evaluate_spd(train_data, spd)
        mean_test_score = np.mean(result_dict["test_score"])
        self.assertTrue(0.45 < mean_test_score < 0.55)
        logging.info(f'Cross validation: {result_dict["test_score"]} with mean: {mean_test_score}')


    def test_eval_spd_affine_invariant(self):
        n_samples = 10000
        n_features = 16
        binary_lr_on_spd = BinaryLRManifold(n_features, n_samples)
        train_data = binary_lr_on_spd.generate_random_data()
        logging.info(f'Training data shape: {train_data.shape()}')
        spd = binary_lr_on_spd.create_spd(SPDAffineMetric)
        result_dict = binary_lr_on_spd.evaluate_spd(train_data, spd)
        mean_test_score = np.mean(result_dict["test_score"])
        self.assertTrue(0.45 < mean_test_score < 0.55)
        logging.info(f'Cross validation: {result_dict["test_score"]} with mean: {mean_test_score}')

    def test_eval_euclidean(self):
        n_samples = 6000
        n_features = 16
        binary_lr_on_spd = BinaryLRManifold(n_features, n_samples)
        train_data = binary_lr_on_spd.generate_random_data()
        logging.info(f'Training data shape: {train_data.shape()}')
        result_dict = binary_lr_on_spd.evaluate_euclidean(train_data)
        mean_test_score = np.mean(result_dict["test_score"])
        self.assertTrue(0.45 < mean_test_score < 0.55)
        logging.info(f'Cross validation: {result_dict["test_score"]} with mean: {mean_test_score}')


if __name__ == '__main__':
    unittest.main()
