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


# Python standard library imports
from typing import AnyStr, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
# 3rd Party imports
import torch
# Library imports
from play import Play
from play.gnn_training_play import GNNTrainingPlay
from util.monitor_memory_device import monitor_memory_device
import python


@dataclass
class GNNMemoryMonitorConfig:
    target_device: AnyStr
    mixed_precision: Optional[torch.dtype]
    hidden_dimension: int
    checkpoint: bool
    pin_memory: bool
    num_workers: int
    neighbors_sampling: AnyStr
    batch_size: int

    def __str__(self) -> AnyStr:
        return (f'{self.target_device=}, {self.mixed_precision=}, {self.hidden_dimension=}, {self.checkpoint=}, '
                f'{self.pin_memory=}, {self.num_workers=}, {self.neighbors_sampling=}, {self.batch_size=}')

    def asdict(self) -> Dict[AnyStr, Any]:
        return asdict(self)

    def __call__(self, gnn_training_play: GNNTrainingPlay) -> None:
        gnn_training_play.training_attributes['tensor_mix_precision'] = self.mixed_precision
        gnn_training_play.training_attributes['checkpoint_enabled'] = self.checkpoint
        gnn_training_play.sampling_attributes['pin_memory'] = self.pin_memory
        gnn_training_play.sampling_attributes['num_workers'] = self.num_workers
        gnn_training_play.sampling_attributes['batch_size'] = self.batch_size


class GNNMemoryMonitor(Play):
    def __init__(self,
                 gnn_training_play: GNNTrainingPlay,
                 gnn_memory_monitor_config: GNNMemoryMonitorConfig) -> None:
        super(GNNMemoryMonitor, self).__init__()

        self.gnn_memory_monitor_config = gnn_memory_monitor_config
        self.gnn_training_play = gnn_training_play
        # Step 4: Update the memory monitor configuration
        self.gnn_memory_monitor_config(self.gnn_training_play)

    @monitor_memory_device
    def play(self) -> None:
        # Step 1: Retrieve the evaluation model
        this_model, class_weights = self.gnn_training_play.get_eval_model()

        # Step 2: Retrieve the training environment
        gnn_training = self.gnn_training_play.get_training_env(this_model, class_weights)

        # Step 3: Retrieve the training and validation data loader
        train_loader, val_loader = self.gnn_training_play.get_loaders()

        gnn_training.train(neural_model=this_model,
                           train_loader=train_loader,
                           val_loader=val_loader)


if __name__ == '__main__':
    gnn_memory_monitor_conf = GNNMemoryMonitorConfig(target_device='cpu',
                                                     mixed_precision=None,
                                                     hidden_dimension=64,
                                                     checkpoint=False,
                                                     pin_memory=True,
                                                     num_workers=6,
                                                     neighbors_sampling="NodeNeighbors",
                                                     batch_size=128)
    logging.info(gnn_memory_monitor_conf)
    logging.info(gnn_memory_monitor_conf.asdict())

    training_attributes = {
        'target_device': gnn_memory_monitor_conf.target_device,
        'dataset_name': 'Cora',
        # Model training Hyperparameters
        'learning_rate': 0.0005,
        'weight_decay': 1e-4,
        'momentum': 0.90,
        'batch_size': 64,
        'loss_function': None,
        'encoding_len': -1,
        'train_eval_ratio': 0.9,
        'epochs': 20,
        'weight_initialization': 'Kaiming',
        'optim_label': 'adam',
        'drop_out': 0.25,
        'is_class_imbalance': True,
        'class_weights': None,
        'patience': 2,
        'min_diff_loss': 0.02,
        'hidden_channels': gnn_memory_monitor_conf.hidden_dimension,
        'tensor_mix_precision': gnn_memory_monitor_conf.mixed_precision,
        'checkpoint_enabled': gnn_memory_monitor_conf.checkpoint,
        # Performance metric definition
        'metrics_list': ['Accuracy', 'Precision', 'Recall', 'F1', 'AuROC', 'AuPR'],
        'plot_parameters': {
            'count': 0,
            'title': 'MyTitle',
            'x_label_size': 12,
            'plot_filename': 'myfile'
        }
    }
    sampling_attributes = {
        'id': 'NeighborLoader',
        'num_neighbors': [4],
        'batch_size': 64,
        'replace': True,
        'pin_memory': False,
        'num_workers': 4
    }

    gnn_train_play = GNNTrainingPlay(training_attributes=training_attributes,
                                     sampling_attributes=sampling_attributes)
    gnn_memory_monitor = GNNMemoryMonitor(gnn_train_play, gnn_memory_monitor_conf)
    gnn_memory_monitor.play()



