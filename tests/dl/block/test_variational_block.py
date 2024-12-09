import unittest
import torch
from dl.block.variational_block import VariationalBlock


class VariationalBlockTest(unittest.TestCase):

    def test_init(self):
        hidden_dim = 8
        latent_size = 6
        variational_block = VariationalBlock(hidden_dim, latent_size)
        print(repr(variational_block))
        self.assertTrue(variational_block.in_features() == hidden_dim)

    def test_re_parameterize(self):
        mu = torch.Tensor([1.5, 2.6])
        log_var = torch.Tensor([0.1, 0.6])
        hidden_dim = 8
        latent_size = 2
        variational_block = VariationalBlock(hidden_dim, latent_size)
        new_parameters = variational_block.re_parameterize(mu, log_var)
        self.assertTrue(variational_block.in_features() == hidden_dim)
        print(f'First new params: {str(new_parameters)}')
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        z = mu + std*eps
        print(f'Second new params: {str(z)}')



if __name__ == '__main__':
    unittest.main()