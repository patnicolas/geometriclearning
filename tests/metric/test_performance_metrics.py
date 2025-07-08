import unittest
from metric.performance_metrics import PerformanceMetrics
from metric.metric_type import MetricType
import logging
from metric import MetricException
import os
import python
from python import SKIP_REASON

class PerformanceMetricsTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_add_metric(self):
        performance_metrics = PerformanceMetrics({})
        performance_metrics.register_metric(metric_type=MetricType.Accuracy, encoding_len=-1, is_weighted=False)
        performance_metrics.register_metric(metric_type=MetricType.Precision, encoding_len=-1, is_weighted=False)

        logging.info(performance_metrics.show_registered_metrics())
        self.assertTrue(len(performance_metrics) == 2)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_build(self):
        metrics_list = ['Accuracy', 'Recall']
        is_class_imbalance = False
        performance_metrics = PerformanceMetrics.build(metrics_list, is_class_imbalance)
        logging.info(performance_metrics.show_registered_metrics())
        self.assertTrue(len(performance_metrics) == 3)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_update(self):
        try:
            import numpy as np
            from metric.built_in_metric import BuiltInMetric

            metrics = {
                MetricType.Accuracy: BuiltInMetric(metric_type=MetricType.Accuracy),
                MetricType.Precision: BuiltInMetric(metric_type=MetricType.Precision),
                MetricType.Recall: BuiltInMetric(metric_type=MetricType.Recall),
                MetricType.EvalLoss: BuiltInMetric(metric_type=MetricType.EvalLoss)
            }
            performance_metrics = PerformanceMetrics(metrics)
            logging.info(performance_metrics.show_registered_metrics())

            np_predictions = [np.array(1.0), np.array(0.0), np.array(0.0), np.array(1.0), np.array(1.0), np.array(1.0)]
            np_labels = [np.array(1.0), np.array(1.0), np.array(1.0), np.array(1.0), np.array(1.0), np.array(0.0)]
            val_loss = [2.3, 1.9, 1.8, 1.6, 1.2, 0.9]

            np_pred = np.stack(np_predictions, axis=0)
            np_lab = np.stack(np_labels)
            performance_metrics.update_perf_metrics(np_pred, np_lab)
            performance_metrics.update_metric(MetricType.EvalLoss, sum(val_loss)/len(val_loss))
            logging.info(f'\n{performance_metrics=}')
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except MetricException as _:
            self.assertTrue(False)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_plot_summary(self):
        try:
            import numpy as np
            from metric.built_in_metric import BuiltInMetric

            metrics = {
                MetricType.Accuracy: BuiltInMetric(metric_type=MetricType.Accuracy),
                MetricType.Precision: BuiltInMetric(metric_type=MetricType.Precision),
                MetricType.Recall: BuiltInMetric(metric_type=MetricType.Recall)
            }
            performance_metrics = PerformanceMetrics(metrics)
            logging.info(performance_metrics.show_registered_metrics())

            np_predictions = [np.array(1.0), np.array(0.0), np.array(0.0), np.array(1.0), np.array(1.0), np.array(1.0)]
            np_labels = [np.array(1.0), np.array(1.0), np.array(1.0), np.array(1.0), np.array(1.0), np.array(0.0)]

            np_pred = np.stack(np_predictions, axis=0)
            np_lab = np.stack(np_labels)
            performance_metrics.update_perf_metrics(np_pred, np_lab)
            performance_metrics.update_perf_metrics(np_pred, np_lab)
            logging.info(f'\n{performance_metrics=}')

            output_file_name = 'results'
            performance_metrics.plot_summary(output_file_name)
            self.assertTrue(True)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except MetricException as _:
            self.assertTrue(False)


