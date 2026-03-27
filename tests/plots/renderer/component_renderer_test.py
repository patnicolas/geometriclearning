import unittest
import logging
import python
from plots.renderer.component_renderer import TextRenderer, CommentRenderer
import matplotlib.pyplot as plt

class ComponentRendererTest(unittest.TestCase):

    def test_title_renderer(self):
        try:
            title_renderer = TextRenderer(text='My title',
                                          font_size=16,
                                          font_weight='bold',
                                          font_color='blue',
                                          font_family='sans serif')
            logging.info(title_renderer)
            title_renderer(arg='title')
            plt.show()
            self.assertTrue(True)
        except (ValueError, NotImplementedError) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_xlabel_renderer(self):
        try:
            xlabel_renderer = TextRenderer(text='My X axis',
                                           font_size=16,
                                           font_weight='bold',
                                           font_color='blue',
                                           font_family='sans serif')
            logging.info(xlabel_renderer)
            xlabel_renderer(arg='xlabel')
            plt.show()
            self.assertTrue(True)
        except (ValueError, NotImplementedError) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_comment_renderer(self):
        try:
            comment_renderer = CommentRenderer(text='My comments',
                                               font_size=13,
                                               font_weight='regular',
                                               font_color='red',
                                               font_family='sans serif',
                                               position=(0.5, 0.5))
            logging.info(comment_renderer)
            comment_renderer()
            plt.show()
            self.assertTrue(True)
        except (ValueError, NotImplementedError) as e:
            logging.error(e)
            self.assertTrue(False)

