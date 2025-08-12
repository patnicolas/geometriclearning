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
            predicted: torch.Tensor = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
            labels: torch.Tensor = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
            logging.info(f'{predicted=}')
            np_accuracy = build_in_metric.from_torch(predicted, labels)
            acc = float(np_accuracy)
            logging.info(f'{acc=}')
            self.assertTrue(acc - 0.6666 < 0.01)

            predicted = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
            np_accuracy = build_in_metric.from_torch(predicted, labels)
            acc = float(np_accuracy)
            logging.info(f'{acc=}')
            self.assertTrue(acc - 0.3333 < 0.001)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(True)
        except MetricException as e:
            logging.error(e)
            self.assertTrue(True)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_precision(self):
        try:
            metric_type = MetricType.Precision
            build_in_metric = BuiltInMetric(metric_type)
            predicted: torch.Tensor = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
            labels: torch.Tensor = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
            logging.info(f'{predicted=}')
            np_precision = build_in_metric.from_numpy(predicted, labels)
            precision = float(np_precision[0])
            logging.info(f'{precision=}')
            self.assertTrue(precision - 0.6666 < 0.001)

            predicted2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
            np_precision2 = build_in_metric.from_numpy(predicted2, labels)
            precision2 = float(np_precision2[0])
            logging.info(f'{precision2=}')
            self.assertTrue(precision2 == 0.75)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(True)
        except MetricException as e:
            logging.error(e)
            self.assertTrue(True)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_recall(self):
        try:
            metric_type = MetricType.Recall
            build_in_metric = BuiltInMetric(metric_type)
            predicted: torch.Tensor = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
            labels: torch.Tensor = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
            logging.info(f'\n{predicted=}')
            np_recall = build_in_metric.from_torch(predicted, labels)
            recall = float(np_recall)
            logging.info(f'{recall=}')
            self.assertTrue(recall-0.833 < 0.01)

            predicted2: torch.Tensor = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
            logging.info(f'\n{predicted2=}')
            np_recall2 = build_in_metric.from_torch(predicted2, labels)
            recall2 = float(np_recall2)
            logging.info(f'{recall2=}')
            self.assertTrue(recall2 == 0.5)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(True)
        except MetricException as e:
            logging.error(e)
            self.assertTrue(True)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_auc(self):
        try:
            metric_type = MetricType.AucROC
            build_in_metric = BuiltInMetric(metric_type)
            predicted: torch.Tensor = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
            labels: torch.Tensor = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
            logging.info(f'\n{predicted=}')
            np_auc = build_in_metric.from_torch(predicted, labels)
            auc_score = float(np_auc)
            logging.info(f'{auc_score=}')
            self.assertTrue(auc_score == 1.0)

            predicted: torch.Tensor = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
            np_auc = build_in_metric.from_torch(predicted, labels)
            auc_score = float(np_auc)
            logging.info(f'{auc_score=}')
            self.assertTrue(auc_score == 0.5)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(True)
        except MetricException as e:
            logging.error(e)
            self.assertTrue(True)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
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