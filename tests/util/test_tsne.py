from unittest import TestCase
import torch
from util.tsne import T_SNE
import logging
import util


class TestT_SNE(TestCase):

    def test_forward_2(self):
        try:
            fig_save = '../../images/tsne2.png'
            x = torch.rand((50, 4))
            n_components = 2
            t_sne = T_SNE(n_components, "inferno_r", fig_save, "This plot")
            y = t_sne.forward(x)
            logging.info(y)
        except Exception as e:
            self.fail(str(e))


    def test_forward_3(self):
        try:
            fig_save = '../../images/tsne3.png'
            n_components = 3
            x = torch.rand((50, 4))
            t_sne = T_SNE(n_components, "inferno_r", fig_save, "This plot")
            y = t_sne.forward(x)
            logging.info(y)
        except Exception as e:
            self.fail(str(e))
