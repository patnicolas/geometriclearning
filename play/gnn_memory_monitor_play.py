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
from typing import AnyStr, Dict, Any, Optional, List, Tuple
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
    """
    Configuration data class for the training and model parameters used in the optimization of memory.
    Note: The following parameters are a subset of training and model configuration that have an impact on the
    consumption of memory on GPU during training.

    @param target_device Target device ('CPU', 'CUDA', 'MPS') informational only
    @type target_device AnyStr
    @param mixed_precision: Optional floating point data type (torch.float64, torch.float32 and torch.float16) for
                            representation of the model parameters
    @type mixed_precision: torch.dtype
    @param hidden_dimension: Number of units in the hidden layer
    @type hidden_dimension: int
    @param checkpoint: Flag for activation checkpointing
    @type checkpoint: bool
    @param pin_memory: Flag for asynchronous data transfer from CPU to GPU (target device)
    @type pin_memory: bool
    @param num_workers: Number of workers used in the Graph data loader
    @type num_workers: int
    @param neighbors_sampling: Type of PyTorch Geometric graph data loader - node neighbors sampler
    @type neighbors_sampling: str
    @param batch_size: Size of the batch of nodes used in training
    @type batch_size: int
    """
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
        """
        Converts this configuration data class into a dictionary
        @return: Dictionary version of this configuration
        @rtype: Dict[AnyStr, Any]
        """
        return asdict(self)

    def __call__(self, gnn_training_play: GNNTrainingPlay) -> None:
        """
        Update some of the training attribute for Graph Neural Network with this configuration
        @param gnn_training_play: Training class for Graph Neural Network
        type gnn_training_play:  GNNTrainingPlay
        """
        gnn_training_play.training_attributes['tensor_mix_precision'] = self.mixed_precision
        gnn_training_play.training_attributes['checkpoint_enabled'] = self.checkpoint
        gnn_training_play.sampling_attributes['pin_memory'] = self.pin_memory
        gnn_training_play.sampling_attributes['num_workers'] = self.num_workers
        gnn_training_play.sampling_attributes['batch_size'] = self.batch_size
        gnn_training_play.sampling_attributes['hidden_channels'] = self.hidden_dimension


class GNNMemoryMonitor(Play):
    """
    Source code related to the Substack article 'Slimming the Graph Neural Network Footprint'
    Ref:

    Class to monitor and report the memory consumption on target device/GPU during training.
    This class leverage the GNNTrainingPlay used for the substack article 'Plug and Play Training of Graph
    Convolutional Network' (https://patricknicolas.substack.com/p/plug-and-play-training-for-graph)
    with the implementation on Github repo:
    https://github.com/patnicolas/geometriclearning/blob/main/play/gnn_training_play.py

    The memory consumption is collected through a decorator defined in util/monitor_memory_device.py

    """
    def __init__(self,
                 gnn_training_play: GNNTrainingPlay,
                 gnn_memory_monitor_config: GNNMemoryMonitorConfig) -> None:
        """
        Constructor for the class to monitor/record memory consumption of a graph convolutional network during
        training given a training & model configuration (GNNTrainingPlay) and set of configuration parameters
        (GNNMemoryMonitorConfig)

        @param gnn_training_play: Training class for Graph Convolutional Network
        @type gnn_training_play: GNNTrainingPlay
        @param gnn_memory_monitor_config: Instance of data class for the configuration parameters used in recording
                    the memory consumption of the GCN model during training
        @type gnn_memory_monitor_config: GNNMemoryMonitorConfig
        """
        super(GNNMemoryMonitor, self).__init__()

        self.gnn_memory_monitor_config = gnn_memory_monitor_config
        self.gnn_training_play = gnn_training_play
        # Update the memory monitor configuration
        self.gnn_memory_monitor_config(self.gnn_training_play)

    @monitor_memory_device
    def play(self) -> None:
        # Step 1: Retrieve the evaluation model
        this_model, class_weights = self.gnn_training_play.get_eval_model()

        # Step 2: Retrieve the training environment
        gnn_training = self.gnn_training_play.get_training_env(this_model, class_weights)

        # Step 3: Retrieve the training and validation data loader
        train_loader, val_loader = self.gnn_training_play.get_loaders()

        # Step 4: Train models
        gnn_training.train(neural_model=this_model, train_loader=train_loader, val_loader=val_loader)

    @staticmethod
    def plot_memory_usage(filenames: List[Tuple[AnyStr, AnyStr]]) -> None:
        all_values = []
        all_labels = []
        for filename, desc in filenames:
            all_values.append(GNNMemoryMonitor.__single_memory_usage(filename))
            all_labels.append(desc)

        from plots.plotter import PlotterParameters, Plotter
        plot_parameters_dict = {
            'count': 0,
            'x_label': 'Batch count',
            'y_label': 'Memory usage (KB)',
            'title': 'Impact of Size of Hidden Layers\non Memory Usage',
            'fig_size': (8, 8),
            'multi_plot_pause': 0
        }
        plot_parameters = PlotterParameters.build(plot_parameters_dict)
        Plotter.plot(all_values, all_labels, plot_parameters)

    @staticmethod
    def __single_memory_usage(filename: AnyStr, corrector: float = 1.0) -> List[int]:
        with open(filename, 'rt') as f:
            lines = f.readlines()
            values = [int(line) for line in lines]
        return values[:16]


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

    # Our configurable training attributes
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

    # A specific Graph Data Loader
    sampling_attributes = {
        'id': 'NeighborLoader',
        'num_neighbors': [8, 4],
        'batch_size': 64,
        'replace': True,
        'pin_memory': False,
        'num_workers': 4
    }

    gnn_train_play = GNNTrainingPlay(training_attributes=training_attributes,
                                     sampling_attributes=sampling_attributes)
    gnn_memory_monitor = GNNMemoryMonitor(gnn_train_play, gnn_memory_monitor_conf)
    gnn_memory_monitor.play()




