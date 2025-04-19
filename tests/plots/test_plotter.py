import unittest

from plots.plotter import PlotterParameters, Plotter


class PlotterTest(unittest.TestCase):

    def test_scaling_ticks(self):
        y_lim, x_delta, y_delta = Plotter.arrange_y((23, 2.9))
        print(f'y_lim: {y_lim} x_delta: {x_delta}, y_delta: {y_delta} ')
        assert y_lim == 3, 'should be 3'
        y_lim, x_delta, y_delta = Plotter.arrange_y((9, 0.9))
        print(f'y_lim: {y_lim} x_delta: {x_delta}, y_delta: {y_delta} ')
        assert y_lim == 1, 'should be 1'
        y_lim, x_delta, y_delta = Plotter.arrange_y((13, 1.9))
        print(f'y_lim: {y_lim} x_delta: {x_delta}, y_delta: {y_delta} ')
        assert y_lim == 2, 'should be 2'
        y_lim, x_delta, y_delta = Plotter.arrange_y((60, 4.4))
        print(f'y_lim: {y_lim} x_delta: {x_delta}, y_delta: {y_delta} ')
        assert y_lim == 5, 'should be 5'

    def test_multi_plot(self):
        dict_values = {
            'Precision': [0.1, 0.3, 0.6, 0.65, 0.7, 0.76],
            'Accuracy': [0.3, 0.4, 0.45, 0.49, 0.52, 0.53],
            'TrainLoss': [3.9, 2.7, 1.65, 1.6, 0.56, 0.53],
            'EvalLoss': [1.9, 1.7, 0.95, 0.6, 0.56, 0.53]
        }
        plotter_params_list = [
            PlotterParameters(0, 'epochs', 'Precision', ''),
            PlotterParameters(0, 'epochs', 'Accuracy', ''),
            PlotterParameters(0, 'epochs', 'TrainLoss', ''),
            PlotterParameters(0, 'epochs', 'EvalLoss', ''),
        ]
        Plotter.multi_plot(dict_values, plotter_params_list, 'Test 1')
