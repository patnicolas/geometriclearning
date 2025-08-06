import unittest
from metric.metric_type import MetricType, get_metric_type
from metric import MetricException
import logging
import python

class MetricTypeTest(unittest.TestCase):

    def test_init(self):
        try:
            logging.info(f'F1 name: {MetricType.F1}')
            metric = get_metric_type('Recall')
            logging.info(f'Metric {metric}')
            self.assertTrue(True)
        except MetricException as e:
            logging.error(e)
            self.assertTrue(False)
