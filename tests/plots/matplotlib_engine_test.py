import unittest
import logging
import python
import torch
import numpy as np

from plots.matplotlib_engine import MatplotlibEngine
from plots.plotting_config import PlottingTextConfig, PlottingConfig

class MatplotlibEngineTest(unittest.TestCase):

    def test_init(self):
        try:
            data_dict = {'accuracy': np.array([0.3, 0.5, 0.8]), 'f1': np.array([0.45, 0.56, 0.6])}
            plotting_config = PlottingConfig(plot_type='plot',
                                             title_config=PlottingTextConfig('My title', 16, 'bold', 'blue'),
                                             x_label_config=PlottingTextConfig('X', 13, 'regular', 'black'),
                                             y_label_config=PlottingTextConfig('Y', 13, 'bold', 'black'),
                                             comment_config=PlottingTextConfig('my plot', 10, 'regular', 'red'),
                                             color_palette='deep')
            plotting_engine = MatplotlibEngine(data_dict, plotting_config)
            logging.info(plotting_engine)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_build_single(self):
        try:
            plotting_config = PlottingConfig(plot_type='plot',
                                             title_config=PlottingTextConfig('My title', 16, 'bold', 'blue'),
                                             x_label_config=PlottingTextConfig('X', 13, 'regular', 'black'),
                                             y_label_config=PlottingTextConfig('Y', 13, 'bold', 'black'),
                                             comment_config=PlottingTextConfig('my plot', 10, 'regular', 'red'),
                                             color_palette='deep')
            plotting_engine = MatplotlibEngine.build_single('accuracy', np.array([0.3, 0.5, 0.8]), plotting_config)
            logging.info(plotting_engine)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_build_from_torch(self):
        try:
            data_dict = {'accuracy': torch.Tensor([0.3, 0.5, 0.8]), 'f1': torch.Tensor([0.45, 0.56, 0.6])}
            plotting_config = PlottingConfig(plot_type='line_plot',
                                             title_config=PlottingTextConfig('My title', 16, 'bold', 'blue'),
                                             x_label_config=PlottingTextConfig('X', 13, 'regular', 'black'),
                                             y_label_config=PlottingTextConfig('Y', 13, 'bold', 'black'),
                                             comment_config=PlottingTextConfig(text='Comments',
                                                                               font_size=10,
                                                                               font_weight='regular',
                                                                               font_color='red',
                                                                               position=(0.5, 0.4)),
                                             background_color='blue')
            plotting_engine = MatplotlibEngine.build_from_torch(data_dict, plotting_config)
            logging.info(plotting_engine)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_render_line_plot(self):
        try:
            data_dict = {'x-axis': np.array([0.1, 0.2, 0.3]),
                         'accuracy': np.array([0.3, 0.5, 0.8]),
                         'f1': np.array([0.45, 0.56, 0.6])}
            plotting_config = PlottingConfig(plot_type='line_plots',
                                             title_config=PlottingTextConfig('My title', 16, 'bold', 'blue'),
                                             x_label_config=PlottingTextConfig('X-values', 13, 'regular', 'black'),
                                             y_label_config=PlottingTextConfig('Y-values', 13, 'bold', 'black'),
                                             comment_config=PlottingTextConfig(text='Comments',
                                                                               font_size=16,
                                                                               font_weight='bold',
                                                                               font_color='red',
                                                                               position=(0.4, 0.6)),
                                             background_color='lightgrey',
                                             color_palette='deep')
            plotting_engine = MatplotlibEngine(data_dict, plotting_config)
            logging.info(plotting_engine)
            plotting_engine.render()
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)