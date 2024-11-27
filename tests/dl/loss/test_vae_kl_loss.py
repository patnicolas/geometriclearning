import unittest
import torch
import torch.nn as nn
from dl.loss.vae_kl_loss import VAEKLLoss


class VAEKLLossTest(unittest.TestCase):

    def test_init(self):
        mu = torch.Tensor([1.0, 0.5, 0.25])
        log_var = torch.Tensor([0.4, -0.3, -0.1])
        loss_function = nn.CrossEntropyLoss()
        vae_kl_loss = VAEKLLoss(mu, log_var, 20, loss_function)
        print(repr(vae_kl_loss))

    def test_forward(self):
        import math
        mu = torch.Tensor([1.0, 0.5, 0.25])
        log_var = torch.Tensor([0.4, -0.3, -0.1])
        num_records = 20
        loss_func =  nn.CrossEntropyLoss()
        vae_kl_loss = VAEKLLoss(mu, log_var, num_records, loss_func)

        input = torch.Tensor([0.4, 1.0, 2.4])
        target = torch.Tensor([0.8, 0.2, 0.7])
        actual_loss = vae_kl_loss.forward(input, target)
        print(f'\nActual loss: {actual_loss}')

        kl_loss = (-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())) / num_records
        reconstruction_loss = loss_func(input, target)
        total_loss = reconstruction_loss + kl_loss
        print(f'\nManual computed loss: {total_loss}')
        self.assertTrue( math.fabs(total_loss - actual_loss) < 0.0001)


if __name__ == '__main__':
    unittest.main()
