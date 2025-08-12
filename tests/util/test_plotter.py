import unittest
from plots.plotter import Plotter, PlotterParameters


class PlotterTest(unittest.TestCase):

    def test_plot_1(self):
        import math
        x = [math.sin(0.1*x) for x in range(0, 100)]
        plotter_params = PlotterParameters(0, 'X', 'Y-sin', 'Plot sine')
        Plotter.single_plot(x, plotter_params)

    @unittest.skip('ignore')
    def test_plot(self):
        import math

        plotter_params = PlotterParameters(0, 'X', 'Y', 'Comparison', (12, 8))
        labels = ['sin', 'sqrt', 'exp']
        x = [math.sin(0.1 * x) for x in range(0, 100)]
        y = [0.1 * math.sqrt(x) for x in range(0, 100)]
        z = [math.exp(-x * 0.1) for x in range(0, 100)]
        Plotter.plot([x, y, z], labels, plotter_params)


if __name__ == '__main__':
    unittest.main()
