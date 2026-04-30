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
import torch
from torch.optim import Optimizer

__all__ = ['SGLD']

class SGLD(Optimizer):
    def __init__(self, params, lr=1e-2, add_noise=True):
        defaults = dict(lr=lr, add_noise=add_noise)
        super(SGLD, self, ).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get parameters
                dw = p.grad.data
                lr = group['lr']

                # Standard SGD update
                p.data.add_(dw, alpha=-lr)

                # Add Langevin Noise: sqrt(2 * lr) * N(0,1)
                if group['add_noise']:
                    noise = torch.randn_like(p.data) * torch.sqrt(torch.tensor(2.0 * lr))
                    p.data.add_(noise)