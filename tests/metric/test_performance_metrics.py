
import unittest
from metric.performance_metrics import PerformanceMetrics
from metric.metric_type import MetricType
import logging

class PerformanceMetricsTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_add_metric(self):
        performance_metrics = PerformanceMetrics({})
        performance_metrics.add_metric(metric_label=MetricType.Accuracy, encoding_len=-1,is_weighted=False)
        performance_metrics.add_metric(metric_label=MetricType.Precision, encoding_len=-1, is_weighted=False)

        logging.info(performance_metrics.show_metrics())
        self.assertTrue(len(performance_metrics) == 2)

    @unittest.skip('Ignored')
    def test_build(self):
        attributes = {'Accuracy': True, 'Recall': True}
        performance_metrics = PerformanceMetrics.build(attributes)
        logging.info(performance_metrics.show_metrics())
        self.assertTrue(len(performance_metrics) == 2)

    @unittest.skip('Ignored')
    def test_update(self):
        import numpy as np
        from metric.built_in_metric import BuiltInMetric

        metrics = {
            MetricType.Accuracy: BuiltInMetric(metric_type=MetricType.Accuracy),
            MetricType.Precision: BuiltInMetric(metric_type=MetricType.Precision),
            MetricType.Recall: BuiltInMetric(metric_type=MetricType.Recall)
        }
        performance_metrics = PerformanceMetrics(metrics)
        logging.info(performance_metrics.show_metrics())

        np_predictions = [np.array(1.0), np.array(0.0), np.array(0.0), np.array(1.0), np.array(1.0), np.array(1.0)]
        np_labels = [np.array(1.0), np.array(1.0), np.array(1.0), np.array(1.0), np.array(1.0), np.array(0.0)]
        val_loss = [2.3, 1.9, 1.8, 1.6, 1.2, 0.9]

        np_pred = np.stack(np_predictions, axis=0)
        np_lab = np.stack(np_labels)
        performance_metrics.update_performance_values(np_pred, np_lab)
        performance_metrics.update_metric(MetricType.EvalLoss, sum(val_loss)/len(val_loss))
        logging.info(f'Performance:\n{str(performance_metrics)}')

    def test_summary(self):
        import numpy as np
        from metric.built_in_metric import BuiltInMetric

        metrics = {
            MetricType.Accuracy: BuiltInMetric(metric_type=MetricType.Accuracy),
            MetricType.Precision: BuiltInMetric(metric_type=MetricType.Precision),
            MetricType.Recall: BuiltInMetric(metric_type=MetricType.Recall)
        }
        performance_metrics = PerformanceMetrics(metrics)
        logging.info(performance_metrics.show_metrics())

        np_predictions = [np.array(1.0), np.array(0.0), np.array(0.0), np.array(1.0), np.array(1.0), np.array(1.0)]
        np_labels = [np.array(1.0), np.array(1.0), np.array(1.0), np.array(1.0), np.array(1.0), np.array(0.0)]

        np_pred = np.stack(np_predictions, axis=0)
        np_lab = np.stack(np_labels)
        performance_metrics.update_performance_values(np_pred, np_lab)
        performance_metrics.update_performance_values(np_pred, np_lab)
        logging.info(f'Performance:\n{str(performance_metrics)}')

        output_file_name = 'results'
        performance_metrics.summary(output_file_name)

