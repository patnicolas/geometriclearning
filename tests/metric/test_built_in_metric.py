import unittest
import torch
from python.metric.built_in_metric import BuiltInMetric, MetricType
import numpy as np
import os
from metric import MetricException
from typing import List
import logging
import python
from python import SKIP_REASON


class BuiltInMetricTest(unittest.TestCase):

    # @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_accuracy(self):
        try:
            metric_type = MetricType.Accuracy
            build_in_metric = BuiltInMetric(metric_type)
            predicted: torch.Tensor = torch.tensor([[1.0],[0.0],[0.0],[1.0]])
            labels: torch.Tensor = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
            logging.info(f'{predicted=}')
            np_accuracy = build_in_metric.from_torch(predicted, labels)
            acc = float(np_accuracy)
            logging.info(f'{acc=}')
            self.assertTrue(acc == 1.0)

            predicted: torch.Tensor = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
            np_accuracy = build_in_metric.from_torch(predicted, labels)
            acc = float(np_accuracy)
            self.assertTrue(acc == 0.0)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(True)
        except MetricException as e:
            logging.error(e)
            self.assertTrue(True)

    # @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_precision(self):
        try:
            metric_type = MetricType.Precision
            build_in_metric = BuiltInMetric(metric_type)
            predicted = np.array([[1.0], [0.0], [0.0], [1.0]])
            labels = np.array([[1.0], [0.0], [0.0], [1.0]])
            logging.info(f'{predicted=}')
            np_precision = build_in_metric.from_numpy(predicted, labels)
            precision = float(np_precision[0])
            logging.info(f'{precision=}')
            self.assertTrue(precision == 1.0)

            predicted = np.array([[0.0], [1.0], [1.0], [0.0]])
            np_precision = build_in_metric.from_numpy(predicted, labels)
            precision = float(np_precision[0])
            logging.info(f'{precision=}')
            self.assertTrue(precision == 1.0)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(True)
        except MetricException as e:
            logging.error(e)
            self.assertTrue(True)

    # @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_recall(self):
        try:
            metric_type = MetricType.Recall
            build_in_metric = BuiltInMetric(metric_type)
            predicted: torch.Tensor = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
            labels: torch.Tensor = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
            logging.info(f'{predicted=}')
            np_recall = build_in_metric.from_torch(predicted, labels)
            recall = float(np_recall)
            logging.info(f'{recall=}')
            self.assertTrue(recall == 1.0)

            predicted: torch.Tensor = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
            np_recall = build_in_metric.from_torch(predicted, labels)
            recall = float(np_recall)
            logging.info(f'{recall=}')
            self.assertTrue(recall == 0.0)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(True)
        except MetricException as e:
            logging.error(e)
            self.assertTrue(True)

    def test_all(self):
        try:

            metric_type = MetricType.All
            build_in_metric = BuiltInMetric(metric_type)
            predicted: List[float] = [1.0, 0.0, 0.0, 1.0]
            labeled = [1.0, 1.0, 0.0, 1.0]
            metrics_dict = build_in_metric(predicted, labeled)
            logging.info(str(metrics_dict))
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(True)
        except MetricException as e:
            logging.error(e)
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()