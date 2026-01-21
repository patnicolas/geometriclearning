import unittest
import logging
import json
import python
from plots.plotting_config import PlottingTextConfig, PlottingConfig


class PlottingConfigTest(unittest.TestCase):

    def test_init_plotting_text_config(self):
        try:
            plotting_text_config = PlottingTextConfig('My text', 13, 'bold', 'blue')
            logging.info(plotting_text_config)
            json_config_str = plotting_text_config.to_json()
            logging.info(json_config_str)
            plotting_text_config_2 = PlottingTextConfig.build(json_config_str)
            self.assertEqual(plotting_text_config, plotting_text_config_2)
        except (KeyError, ValueError) as e:
            logging.error(e)
            self.assertFalse(True)

    def test_init_plotting_config(self):
        try:
            plotting_title_config = PlottingTextConfig('My title', 16, 'bold', 'blue')
            plotting_x_label_config = PlottingTextConfig('X', 13, 'regular', 'black')
            plotting_y_label_config = PlottingTextConfig('Y', 13, 'bold', 'black')

            plotting_config = PlottingConfig(plot_type='plot',
                                             title_config=plotting_title_config,
                                             x_label_config=plotting_x_label_config,
                                             y_label_config=plotting_y_label_config,
                                             color_palette='deep')
            logging.info(f'\nOriginal:{plotting_config}')
            json_config_str = plotting_config.to_json()
            logging.info(f'\nTo JSON:\n{json_config_str}')
            plotting_config_2 = PlottingConfig.build(json_config_str)
            logging.info(f'\nLoaded:{plotting_config_2}')
            # self.assertEqual(plotting_config, plotting_config_2)
        except (KeyError, ValueError, json.decoder.JSONDecodeError) as e:
            logging.error(e)
            self.assertFalse(True)
