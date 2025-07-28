__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

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


import torch
from ray import tune
from ray.tune import Trainable
from ray.tune import ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
import os
__all__ = ['RayHyperParams']


class RayHyperParams(object):
    def __init__(self, config):
        self.learning_rate_config = config['learning_rate']
        self.batch_size = config['batch_size']
        self.momentum = config['momentum']


class RayTune(object):
    def __init__(self, config: dict, max_num_epochs: int):
        self.config = config
        self.scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)

    def execute(self, data_dir: str, num_samples: int, train_func) -> ExperimentAnalysis:
        return tune.run(
            tune.with_parameters(train_func, data_dir=data_dir),
            resources_per_trial={"cpu": 2, "gpu": 1},
            config= self.config,
            metric="loss",
            mode="min",
            num_samples=num_samples,
            scheduler=self.scheduler
        )


class RayTuneModel(Trainable):
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader, neural_net = NeuralNet):
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._neural_net = neural_net

    def setup(self, config):
        self.train_loader = self._train_loader
        self.test_loader = self._test_loader
        self.model = self._neural_net
        self.optimizer = self._neural_net.hyper_params.optimize_tune(self._neural_net, config)

    def reset_config(self, new_config):
        for param_group in self.optimizer.param_groups:
            if "learning_rate" in new_config:
                param_group["lr"] = new_config["lr"]
            if "momentum" in new_config:
                param_group["momentum"] = new_config["momentum"]
        self.model = self._neural_net
        self.config = new_config
        return True

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "encoder_model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "encoder_model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))

    def step(self):
        ""


