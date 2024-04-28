import unittest
from python.metric.accuracymetric import AccuracyMetric
import numpy as np
import torch


class AccuracyMetricTest(unittest.TestCase):

    def test_from_numpy(self):
        accuracy_metric = AccuracyMetric(0.002)
        predicted = np.array([0.561, 0.525, 0.672, 0.412])
        labels = np.array([0.561, 0.543, 0.671, 0.488])
        accuracy_metric.from_numpy(predicted, labels)
        print(f'accuracy: {accuracy_metric()}')
        assert accuracy_metric() == 0.5

    def test_from_tensor(self):
        accuracy_metric = AccuracyMetric(0.002)
        predicted = torch.Tensor([[1.0], [0.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])
        labels = torch.Tensor([[1.0], [1.0], [0.0], [1.0], [1.0], [1.0], [0.0], [1.0], [0.0]])
        accuracy_metric.from_tensor(predicted, labels)
        print(f'accuracy: {accuracy_metric()}')
        data_size = len(labels)
        assert accuracy_metric() == 1.0 - 2.0/len(labels)


if __name__ == '__main__':
    unittest.main()
