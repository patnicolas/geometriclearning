__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard Library imports
from typing import Optional, Callable
# PyTorch imports
import torch
from torch.optim.optimizer import params_t
from torch.optim import Optimizer

__all__ = ['SGLD']



class SGLD(Optimizer):
    """
        Implementation of the Stochastic Gradient Langevin Dynamics algorithm as a PyTorch Optimizer
            w <- w - lr.grad(L) + sqrt(2.lr).N(0, 1)
        lr: Learning rate
        N: Normal distribution
    """
    def __init__(self, params: params_t, lr: float = 1e-2) -> None:
        """
        Constructor for the Stochastic Gradient Langevin Dynamics.
        @param params: Optimizer configuration parameters
        @type params: params_t
        @param lr:  Learning rate
        @type lr: float
        """
        defaults = dict(lr=lr, add_noise=True)
        super(SGLD, self, ).__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Implementation of the step for the Stochastic Gradient Langevin Dynamics.

        @param closure: reevaluates the model and returns the loss, None in this case
        @type closure: Callable
        @return: Value of the loss (optional) after this step
        @rtype: Optional[float]
        """
        def langevin_update(p: torch.Tensor) -> None:
            # Stochastic Gradient update formula w <- w - alpha.dw
            p.data.add_(p.grad.data, alpha=-self.defaults['lr'])
            # Add Langevin Noise: sqrt(2 * lr) * N(0,1)
            p.data.add_(SGLD.langevin_noise(p.data, self.defaults['lr']))
        [langevin_update(p) for group in self.param_groups for p in group['params']]

    @staticmethod
    def langevin_noise(data: torch.Tensor, lr: float) -> torch.Tensor:
        """
        Define the Gaussian noise component of the loss function. This component has to be
        added to the standard gradient descent.

        @param data: Input data
        @type data: Torch.Tensor
        @param lr: Learning rate
        @type lr: float
        @return: The Gaussian noise component
        @rtype: Torch.Tensor
        """
        return torch.randn_like(data) * torch.sqrt(torch.tensor(lr))

