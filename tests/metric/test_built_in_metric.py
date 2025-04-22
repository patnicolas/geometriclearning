import unittest
import torch
from python.metric.built_in_metric import BuiltInMetric, MetricType
import numpy as np


class BuiltInMetricTest(unittest.TestCase):

    def test_puzzle(self):
        fruit_stores = {
            'apple': {'store1': 1.0, 'store2': 2.0},
            'banana': {'store1': 3.0, 'store2': 2.0},
        }
        result = [{fruit: min(stores, key=stores.get)} for fruit, stores in fruit_stores.items()]
        print(f'Result: {result}')

    def test_accuracy(self):
        metric_type = MetricType.Accuracy
        build_in_metric = BuiltInMetric(metric_type)
        predicted: torch.Tensor = torch.tensor([[1.0],[0.0],[0.0],[1.0]])
        labels: torch.Tensor = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
        print(f'Predicted {str(predicted)}')
        np_accuracy = build_in_metric.from_torch(predicted, labels)
        print(f'Accuracy {np_accuracy}')
        self.assertTrue(float(np_accuracy) == 1.0)

        predicted: torch.Tensor = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        np_accuracy = build_in_metric.from_torch(predicted, labels)
        self.assertTrue(float(np_accuracy) == 0.0)

    def test_precision(self):
        metric_type = MetricType.Precision
        build_in_metric = BuiltInMetric(metric_type)
        predicted = np.array([[1.0], [0.0], [0.0], [1.0]])
        labels = np.array([[1.0], [0.0], [0.0], [1.0]])
        print(f'Predicted {str(predicted)}')
        np_precision = build_in_metric.from_numpy(predicted, labels)
        print(f'Precision: {np_precision}')
        self.assertTrue(float(np_precision) == 1.0)

        predicted = np.array([[0.0], [1.0], [1.0], [0.0]])
        np_precision = build_in_metric.from_numpy(predicted, labels)
        print(f'Precision: {np_precision}')
        self.assertTrue(float(np_precision) == 0.0)

    def test_recall(self):
        metric_type = MetricType.Recall
        build_in_metric = BuiltInMetric(metric_type)
        predicted: torch.Tensor = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
        labels: torch.Tensor = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
        print(f'Predicted {str(predicted)}')
        np_recall = build_in_metric.from_torch(predicted, labels)
        print(f'Recall: {np_recall}')
        self.assertTrue(float(np_recall) == 1.0)

        predicted: torch.Tensor = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        np_recall = build_in_metric.from_torch(predicted, labels)
        print(f'Recall: {np_recall}')
        self.assertTrue(float(np_recall) == 0.0)



if __name__ == '__main__':
    unittest.main()