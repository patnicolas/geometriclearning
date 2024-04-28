import unittest
from python.util.plotter import Plotter, PlotterParameters


class PlotterTest(unittest.TestCase):

    def test_plot_1(self):
        import math
        x = [math.sin(0.1*x) for x in range(0, 100)]
        plotter_params = PlotterParameters(0, 'X', 'Y-sin', 'Plot sine')
        Plotter.single_plot(x, plotter_params)

    def test_plot_2(self):
        import math

        x = [math.sin(0.1 * x) for x in range(0, 100)]
        y = [0.03*math.sqrt(x) for x in range(0, 100)]
        plotter_params1 = PlotterParameters(0, 'X', 'Y-sin', 'Plot sine')
        plotter_params2 = PlotterParameters(1, 'X', 'Y-sqrt', 'Plot sqrt')
        Plotter.multi_plot([x, y], [plotter_params1, plotter_params2])

    def test_plot_3(self):
        import math

        x = [math.sin(0.1 * x) for x in range(0, 100)]
        y = [0.03*math.sqrt(x) for x in range(0, 100)]
        z = [math.exp(-x*0.1) for x in range(0, 100)]

        plotter_params1 = PlotterParameters(0, 'X', 'Y-sin', 'Plot sine')
        plotter_params2 = PlotterParameters(1, 'X', 'Y-sqrt', 'Plot sqrt')
        plotter_params3 = PlotterParameters(2, 'X', 'Y-sqrt', 'Plot log')
        Plotter.multi_plot([x, y, z], [plotter_params1, plotter_params2, plotter_params3])

    def test_plot_4(self):
        import math

        x = [math.sin(0.1 * x) for x in range(0, 100)]
        y = [0.03*math.sqrt(x) for x in range(0, 100)]
        z = [math.exp(-x*0.1) for x in range(0, 100)]
        t = [x*x for x in range(0, 100)]

        plotter_params1 = PlotterParameters(0, 'X', 'Y-sin', 'Plot sine')
        plotter_params2 = PlotterParameters(1, 'X', 'Y-sqrt', 'Plot sqrt')
        plotter_params3 = PlotterParameters(2, 'X', 'Y-log', 'Plot log')
        plotter_params4 = PlotterParameters(3, 'X', 'Y-sqr', 'Plot sqr')
        Plotter.multi_plot([x, y, z, t], [plotter_params1, plotter_params2, plotter_params3, plotter_params4])

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