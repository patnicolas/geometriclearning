import unittest
import torch
import matplotlib.pyplot as plt
from deeplearning.loss.sgld import SGLD
import numpy as np

class SGLDTest(unittest.TestCase):

    def test_sgld(self):
        def normal_dist(x: np.ndarray) -> np.ndarray:
            return 1.0/np.sqrt(2 * np.pi) * np.exp(-0.5 * x ** 2)

        w = torch.randn(1, requires_grad=True)
        optimizer = SGLD([w], lr=0.1)
        num_samples = 10000

        samples = []

        # 2. Run SGLD
        for i in range(num_samples):
            optimizer.zero_grad()
            # Negative log-posterior of a Standard Normal: 0.5 * theta^2
            loss = 0.5 * torch.pow(w, 2)
            loss.backward()
            optimizer.step()
            samples.append(w.item())

        # 3. Error
        x = np.linspace(-4, 4, num_samples)
        normal_x = normal_dist(x)
        mse = sum([(s - d)*(s - d) for s, d in zip(samples, normal_x)])/len(samples)
        print(f'MSE: {mse} Ratio: {100.0*mse/sum(samples)} %')

        # 4. Visualization
        plt.hist(samples, bins=160, density=True, color='darkgrey', alpha=0.7, label='Stochastic Langevin Samples')
        plt.plot(x, normal_x, 'r', label='Target Posterior')
        plt.legend()
        plt.show()
