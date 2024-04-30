
import unittest
import torch
from python.metric.builtinmetric import BuiltInMetric, MetricType
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class BuiltInMetricTest(unittest.TestCase):
    def test_accuracy(self):
        metric_type = MetricType.Accuracy
        build_in_metric = BuiltInMetric(metric_type)
        predicted: torch.Tensor = torch.tensor([[1.0],[0.0],[0.0],[1.0]])
        labels: torch.Tensor = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
        print(f'Predicted {str(predicted)}')
        np_accuracy = build_in_metric(predicted, labels)
        print(f'Accuracy {np_accuracy}')
        assert float(np_accuracy) == 1.0

        predicted: torch.Tensor = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        np_accuracy = build_in_metric(predicted, labels)
        assert float(np_accuracy) == 0.0

    def test_precision(self):
        metric_type = MetricType.Precision
        build_in_metric = BuiltInMetric(metric_type)
        predicted: torch.Tensor = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
        labels: torch.Tensor = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
        print(f'Predicted {str(predicted)}')
        np_precision = build_in_metric(predicted, labels)
        print(f'Precision: {np_precision}')
        assert float(np_precision) == 1.0

        predicted: torch.Tensor = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        np_precision = build_in_metric(predicted, labels)
        print(f'Precision: {np_precision}')
        assert float(np_precision) == 0.0

    def test_recall(self):
        metric_type = MetricType.Recall
        build_in_metric = BuiltInMetric(metric_type)
        predicted: torch.Tensor = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
        labels: torch.Tensor = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
        print(f'Predicted {str(predicted)}')
        np_recall = build_in_metric(predicted, labels)
        print(f'Recall: {np_recall}')
        assert float(np_recall) == 1.0

        predicted: torch.Tensor = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        np_recall = build_in_metric(predicted, labels)
        print(f'Recall: {np_recall}')
        assert float(np_recall) == 0.0

if __name__ == '__main__':
    unittest.main()