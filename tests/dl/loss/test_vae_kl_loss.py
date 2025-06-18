import unittest
import torch
import torch.nn as nn
from dl.loss.vae_kl_loss import VAEKLLoss
import logging
import python


class VAEKLLossTest(unittest.TestCase):

    def test_init(self):
        latent_dim = 12
        mu = nn.Linear(64 * 7 * 7, latent_dim)
        log_var = nn.Linear(64 * 7 * 7, latent_dim)
        loss_function = nn.CrossEntropyLoss()
        vae_kl_loss = VAEKLLoss(mu, log_var, 20, loss_function)
        logging.info(repr(vae_kl_loss))

    def test_forward(self):
        import math

        latent_dim = 12
        mu = nn.Linear(64 * 7 * 7, latent_dim)
        log_var = nn.Linear(64 * 7 * 7, latent_dim)
        loss_function = nn.CrossEntropyLoss()
        num_records = 200
        vae_kl_loss = VAEKLLoss(mu, log_var, 20, loss_function)
        logging.info(repr(vae_kl_loss))

        input = torch.rand(latent_dim, 64 * 7 * 7)
        target = input
        actual_loss = vae_kl_loss.forward(input, target)
        logging.info(f'\nActual loss: {actual_loss}')

        kl_loss = (-0.5 * torch.sum(1 + log_var(input) - mu(input) ** 2 - log_var(input).exp())) / num_records
        reconstruction_loss = loss_function(input, target)
        total_loss = reconstruction_loss + kl_loss
        logging.info(f'\nManual computed loss: {total_loss}')
        self.assertTrue( math.fabs(total_loss - actual_loss) < 0.01)


if __name__ == '__main__':
    unittest.main()
