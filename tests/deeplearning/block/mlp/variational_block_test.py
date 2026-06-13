import unittest
import torch
from deeplearning.block.mlp.variational_block import VariationalBlock
import logging
import os
from python import SKIP_REASON


class VariationalBlockTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init(self):
        hidden_dim = 8
        latent_size = 6
        variational_block = VariationalBlock(hidden_dim, latent_size)
        logging.info(repr(variational_block))
        self.assertTrue(variational_block.in_features() == hidden_dim)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_re_parameterize(self):
        mu = torch.Tensor([1.5, 2.6])
        log_var = torch.Tensor([0.1, 0.6])
        hidden_dim = 8
        latent_size = 2
        variational_block = VariationalBlock(hidden_dim, latent_size)
        new_parameters = variational_block.re_parameterize(mu, log_var)
        self.assertTrue(variational_block.in_features() == hidden_dim)
        logging.info(f'First new params: {str(new_parameters)}')
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        z = mu + std*eps
        logging.info(f'Second new params: {str(z)}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_non_zero_input(self):
        x = torch.zeros(20)
        has_non_zeros = VariationalBlock.input_has_non_zeros(x)
        self.assertFalse(has_non_zeros)
        x[1] = 0.00001
        has_non_zeros = VariationalBlock.input_has_non_zeros(x)
        self.assertTrue(has_non_zeros)

    def test_forward_1(self):
        t1 = torch.Tensor([9, 0, 1.9, 0.5])
        s1 = t1.shape
        l1 = len(s1)

        t2 = torch.Tensor([[9, 0, 1.9], [0.4, 1.3, 9.0]])
        s2 = t2.shape
        l2 = len(s2)

        hidden_dim = 8
        latent_size = 2
        variational_block = VariationalBlock(hidden_dim, latent_size)
        self.assertTrue(variational_block.in_features() == hidden_dim)
        (z, mu, log_var) = variational_block(torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0]))
        logging.info(z)

if __name__ == '__main__':
    unittest.main()