import unittest
import logging
import python
import torch
import numpy as np

from plots.matplotlib_engine import MatplotlibEngine
from plots.plotting_config import (TextRenderer, PlottingConfig, CommentRenderer, PlotFontDict,
                                   AnnotationRenderer, PlotContext)

class MatplotlibEngineTest(unittest.TestCase):

    def test_init(self):
        try:
            data_dict = {'accuracy': np.array([0.3, 0.5, 0.8]), 'f1': np.array([0.45, 0.56, 0.6])}
            title_config = TextRenderer(text='My title',
                                        font_size=16,
                                        font_weight='bold',
                                        font_color='blue',
                                        font_family='sans serif')
            x_label_config = TextRenderer(text='X',
                                          font_size=12,
                                          font_weight='regular',
                                          font_color='black',
                                          font_family='sans serif')
            y_label_config = TextRenderer(text='X',
                                          font_size=12,
                                          font_weight='regular',
                                          font_color='black',
                                          font_family='sans serif')
            legend = PlotFontDict(font_size=12,
                                  font_weight='regular',
                                  font_color='green',
                                  font_family='sans serif')
            comment_config = CommentRenderer(text='Comments',
                                             font_size=10,
                                             font_weight='regular',
                                             font_color='red',
                                             font_family='sans serif',
                                             position=(0.0, 0.0))

            context = PlotContext(grid=True,
                                  background_color='white',
                                  fig_size=(10, 9))
            plotting_config = PlottingConfig(plot_type='plot',
                                             title_config=title_config,
                                             x_label_config=x_label_config,
                                             y_label_config=y_label_config,
                                             comment_config=comment_config,
                                             plot_context=context,
                                             legend_config=legend)

            plotting_engine = MatplotlibEngine(data_dict, plotting_config)
            logging.info(plotting_engine)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_build_single(self):
        try:
            data_dict = {'accuracy': np.array([0.3, 0.5, 0.8]), 'f1': np.array([0.45, 0.56, 0.6])}
            title_config = TextRenderer(text='My title',
                                        font_size=16,
                                        font_weight='bold',
                                        font_color='blue',
                                        font_family='sans serif')
            x_label_config = TextRenderer(text='X',
                                          font_size=12,
                                          font_weight='regular',
                                          font_color='black',
                                          font_family='sans serif')
            y_label_config = TextRenderer(text='X',
                                          font_size=12,
                                          font_weight='regular',
                                          font_color='black',
                                          font_family='sans serif')
            legend = PlotFontDict(font_size=12,
                                  font_weight='regular',
                                  font_color='green',
                                  font_family='sans serif')
            comment_config = CommentRenderer(text='My comments',
                                             font_size=10,
                                             font_weight='regular',
                                             font_color='red',
                                             font_family='sans serif',
                                             position=(0.0, 0.0))

            context = PlotContext(grid=True,
                                  background_color='white',
                                  fig_size=(10, 9))
            plotting_config = PlottingConfig(plot_type='plot',
                                             title_config=title_config,
                                             x_label_config=x_label_config,
                                             y_label_config=y_label_config,
                                             comment_config=comment_config,
                                             plot_context=context,
                                             legend_config=legend)
            plotting_engine = MatplotlibEngine.build_single('accuracy', np.array([0.3, 0.5, 0.8]), plotting_config)
            logging.info(plotting_engine)
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip("Ignored")
    def test_build_from_torch(self):
        try:
            data_dict = {'accuracy': torch.Tensor([0.3, 0.5, 0.8]), 'f1': torch.Tensor([0.45, 0.56, 0.6])}
            plotting_config = PlottingConfig(plot_type='line_plot',
                                             title_config=TextRenderer('My title', 16, 'bold', 'blue'),
                                             x_label_config=TextRenderer('X', 13, 'regular', 'black'),
                                             y_label_config=TextRenderer('Y', 13, 'bold', 'black'),
                                             comment_config=CommentRenderer(text='Comments',
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
            title_config = TextRenderer(text='My title',
                                        font_size=16,
                                        font_weight='bold',
                                        font_color='blue',
                                        font_family='sans serif')
            x_label_config = TextRenderer(text='X',
                                          font_size=12,
                                          font_weight='regular',
                                          font_color='black',
                                          font_family='sans serif')
            y_label_config = TextRenderer(text='X',
                                          font_size=12,
                                          font_weight='regular',
                                          font_color='black',
                                          font_family='sans serif')
            legend = PlotFontDict(font_size=12,
                                  font_weight='regular',
                                  font_color='green',
                                  font_family='sans serif')
            comment_config = CommentRenderer(text='New comments ..',
                                             font_size=10,
                                             font_weight='regular',
                                             font_color='red',
                                             font_family='sans serif',
                                             position=(0.4, 0.6))
            context = PlotContext(grid=True,
                                  background_color='white',
                                  fig_size=(8, 8))
            annotate_config = AnnotationRenderer(text='My annotation',
                                                 xy=(0, 1),
                                                 xytext=(1, 2),
                                                 arrow_style='->',
                                                 connection_style="arc3,rad=.2",
                                                 color='blue')
            plotting_config = PlottingConfig(plot_type='line_plots',
                                             title_config=title_config,
                                             x_label_config=x_label_config,
                                             y_label_config=y_label_config,
                                             comment_config=comment_config,
                                             annotate_config=annotate_config,
                                             plot_context=context,
                                             legend_config=legend)

            plotting_engine = MatplotlibEngine(data_dict, plotting_config)
            logging.info(plotting_engine)
            plotting_engine.render()
            self.assertTrue(True)
        except ValueError as e:
            logging.error(e)
            self.assertTrue(False)