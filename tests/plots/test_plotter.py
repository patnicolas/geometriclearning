import unittest

from plots.plotter import PlotterParameters, Plotter
import matplotlib.pyplot as plt

class PlotterTest(unittest.TestCase):

    # @unittest.skip('Ignore')
    def test_scatter_point(self):
        homophilies = [0.815, 0.815, 0.815, 0.302, 0.302, 0.302]
        precisions = [0.78, 0.81, 0.84, 0.49, 0.65, 0.88]
        labels = ['Cora-4', 'Cora-10,4', 'Cora-12,6,3', 'Flickr-4', 'Flickr-10,4', 'Flickr-12,6,3']
        colors = ['blue', 'red', 'purple', 'green', 'black', 'orange']

        plt.figure(figsize=(8, 6))
        plt.scatter(homophilies, precisions, s=140, c=colors, alpha=0.8, marker='o')
        offset = -0.02
        ha = 'right'
        for i, label in enumerate(labels):
            if i > 2:
                offset = 0.02
                ha = 'left'
            plt.text(homophilies[i] + offset,
                     precisions[i]-0.006,
                     label,
                     fontdict={'fontsize': 14, 'fontname': 'Helvetica', 'fontweight': 'bold', 'ha': ha})

        plt.title(label='Precision vs. Node+Edge Homophily\nGraph Convolutional Network - Node Neighbor Sampling',
                  fontsize=20,
                  fontname='Helvetica')
        plt.xlabel(xlabel='Homophily ratio',  fontdict={'fontsize': 16, 'fontname': 'Helvetica', 'fontweight': 'bold'})
        plt.ylabel(ylabel='Precision',  fontdict={'fontsize': 16, 'fontname': 'Helvetica', 'fontweight': 'bold'})
        plt.xticks(fontsize=13, fontname='Helvetica')
        plt.yticks(fontsize=13, fontname='Helvetica')
        plt.grid(True)
        plt.show()

    @unittest.skip('Ignore')
    def test_bar_charts(self):
        import numpy as np
        # Sample data
        labels = ['Cora', 'PubMed', 'CiteSeer', 'Wikipedia', 'Flickr']
        node_homophily = [0.83, 0.79, 0.71, 0.11, 0.32]
        edge_homophily = [0.81, 0.80, 0.74, 0.24, 0.32]
        edge_insensitive_homophily = [0.77, 0.66, 0.62, 0.06, 0.07]

        # Positioning for bars
        x_pos = np.arange(len(labels))
        width = 0.40  # Width of each bar

        # Create grouped bar chart
        plt.bar(x_pos - width / 2, node_homophily, width, label='Node Homophily', color='#fa3c36')
        plt.bar(x_pos, edge_homophily, width, label='Edge Homophily', color='#0dc1fa')
        plt.bar(x_pos + width / 2, edge_insensitive_homophily, width, label='Class Insensitive Edge Homophily', color='#a403c4')
        # Add labels and formatting
        plt.xlabel(xlabel='Datasets', fontdict={'fontsize': 16, 'fontname': 'Helvetica', 'fontweight': 'bold'})
        plt.ylabel(ylabel='Homophily Ratio', fontdict={'fontsize': 16, 'fontname': 'Helvetica', 'fontweight': 'bold'})
        plt.title(label='Evaluation Homophily Ratios for Common Graph Datasets', fontsize=22, fontname='Helvetica')
        plt.xticks(x_pos, labels, fontsize=12, fontname='Helvetica')
        plt.yticks(fontsize=12, fontname='Helvetica')
        plt.legend(fontsize=12)

        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    @unittest.skip('Ignore')
    def test_bar_charts_2(self):
        import numpy as np

        # Sample data
        labels = ['[4]', '[4, 2]', '[12, 6, 3]']
        cora = [0.61, 0.78, 0.81]
        flickr = [0.78, 0.71, 0.81]
        # Positioning for bars
        x_pos = np.arange(len(labels))
        width = 0.40  # Width of each bar

        # Create grouped bar chart
        plt.bar(x_pos - width / 2, cora, width, label='Cora', color='#0dc1fa')
        # plt.bar(x_pos, edge_homophily, width, label='Edge Homophily', color='#0dc1fa')
        plt.bar(x_pos + width / 2, flickr, width, label='Flickr',
                color='#a403c4')
        # Add labels and formatting
        plt.xlabel(xlabel='Node Neighbors Distribution', fontdict={'fontsize': 16, 'fontname': 'Helvetica', 'fontweight': 'bold'})
        plt.ylabel(ylabel='F1 Metric', fontdict={'fontsize': 16, 'fontname': 'Helvetica', 'fontweight': 'bold'})
        plt.title(label='Performance of Graph Convolutional Network for Cora and Flickr Datasets',
                  fontsize=18,
                  fontname='Helvetica')
        plt.xticks(x_pos, labels, fontsize=12, fontname='Helvetica')
        plt.yticks(fontsize=12, fontname='Helvetica')
        plt.legend(fontsize=12)

        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    @unittest.skip('Ignore')
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

    @unittest.skip('Ignore')
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
