import unittest
import logging
import json
import python
from plots.plotting_config import TextRenderer, PlottingConfig, PlotFontDict, PlotContext


class PlottingConfigTest(unittest.TestCase):

    def test_init_plotting_text_config(self):
        try:
            text_config = TextRenderer(text='My title',
                                       font_size=16,
                                       font_weight='bold',
                                       font_color='blue',
                                       font_family='sans serif')
            logging.info(text_config)
            json_config_str = text_config.to_json()
            logging.info(json_config_str)
            plotting_text_config_2 = TextRenderer.build(json_config_str)
            self.assertEqual(text_config, plotting_text_config_2)
        except (KeyError, ValueError) as e:
            logging.error(e)
            self.assertFalse(True)

    def test_init_plotting_config(self):
        try:
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
                                  font_color='red',
                                  font_family='sans serif')
            context = PlotContext(grid=True,
                                  background_color='white',
                                  fig_size=(10, 8))
            plotting_config = PlottingConfig(plot_type='plot',
                                             title_config=title_config,
                                             x_label_config=x_label_config,
                                             y_label_config=y_label_config,
                                             plot_context=context,
                                             legend_config=legend)
            logging.info(f'\nOriginal:{plotting_config}')
            json_config_str = plotting_config.to_json()
            logging.info(f'\nTo JSON:\n{json_config_str}')
            plotting_config_2 = PlottingConfig.build(json_config_str)
            logging.info(f'\nLoaded:{plotting_config_2}')
            # self.assertEqual(plotting_config, plotting_config_2)
        except (KeyError, ValueError, json.decoder.JSONDecodeError) as e:
            logging.error(e)
            self.assertFalse(True)
