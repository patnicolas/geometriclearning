from unittest import TestCase

from util.plottermixin import PlotterParameters, PlotterMixin


class TestPlotterMixin(TestCase):
    def test_two_plot(self):
        values_1 = [9.123, 7.123, 4.100, 2.90, 1.0]
        values_2 = [1.0, 0.7123, 0.4100, 0.290, 0.01]
        plot1 = PlotterParameters(len(values_1), "var-1", "var-2", "First plot")
        plot2 = PlotterParameters(len(values_2), "var-3", "var-4", "\nSecond plot")
        PlotterMixin.two_plot(values_1, values_2, [plot1, plot2])
