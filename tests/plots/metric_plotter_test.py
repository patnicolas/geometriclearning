import unittest
import logging
from plots.metric_plotter import MetricPlotterParameters, MetricPlotter
import python


class MetricPlotterTest(unittest.TestCase):

    # @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_metric_parameters(self):
        plotter_params = MetricPlotterParameters(count=0,
                                                 x_label='epochs',
                                                 title='Test plot 1',
                                                 x_label_size=12,
                                                 plot_filename='../output_plots')
        logging.info(plotter_params)
        logging.info(repr(plotter_params))
        self.assertTrue(True)

    # @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_plot_save(self):
        try:
            dict_values = {
                'Precision': [0.1, 0.3, 0.6, 0.65, 0.7, 0.76],
                'Accuracy': [0.3, 0.4, 0.45, 0.49, 0.52, 0.53],
                'TrainLoss': [3.9, 2.7, 1.65, 1.6, 0.56, 0.53],
                'EvalLoss': [1.9, 1.7, 0.95, 0.6, 0.56, 0.53]
            }
            plotter_params = MetricPlotterParameters(count=0,
                                                     x_label='epochs',
                                                     title='Test plot 1',
                                                     x_label_size=12,
                                                     fig_size=(12, 8))
            metric_plot = MetricPlotter(plotter_params=plotter_params)
            metric_plot.plot(dict_values=dict_values)
            self.assertTrue(True)
        except FileNotFoundError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_plot_save_fail(self):
        try:
            dict_values = {
                'Precision': [0.1, 0.3, 0.6, 0.65, 0.7, 0.76],
                'Accuracy': [0.3, 0.4, 0.45, 0.49, 0.52, 0.53],
                'TrainLoss': [3.9, 2.7, 1.65, 1.6, 0.56, 0.53],
                'EvalLoss': [1.9, 1.7, 0.95, 0.6, 0.56, 0.53]
            }
            plotter_params = MetricPlotterParameters(count=0,
                                                     x_label='epochs',
                                                     title='Test plot 1',
                                                     x_label_size=12,
                                                     fig_size=(12, 8))
            metric_plot = MetricPlotter(plotter_params=plotter_params)
            metric_plot.plot(dict_values=dict_values)
            self.assertTrue(False)
        except FileNotFoundError as e:
            logging.error(e)
            self.assertTrue(True)

    def test_plot_display(self):
        try:
            dict_values = {
                'Precision': [0.1, 0.3, 0.6, 0.65, 0.7, 0.76],
                'Accuracy': [0.3, 0.4, 0.45, 0.49, 0.52, 0.53],
                'TrainLoss': [3.9, 2.7, 1.65, 1.6, 0.56, 0.53],
                'EvalLoss': [1.9, 1.7, 0.95, 0.6, 0.56, 0.53]
            }
            plotter_params = MetricPlotterParameters(count=0,
                                                     x_label='epochs',
                                                     title='Test plot 1',
                                                     x_label_size=12,
                                                     fig_size=(12, 8))
            metric_plot = MetricPlotter(plotter_params=plotter_params)
            metric_plot.plot(dict_values=dict_values)
        except FileNotFoundError as e:
            logging.error(e)
            self.assertTrue(True)