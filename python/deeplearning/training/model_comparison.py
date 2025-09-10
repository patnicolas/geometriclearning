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

# Standard library imports
from typing import AnyStr, List, Dict
import logging
# 3rd Party imports
from torch.utils.data import DataLoader
# Library imports
from deeplearning.training.neural_training import NeuralTraining
from deeplearning.model.neural_model import NeuralModel
from metric.performance_metrics import PerformanceMetrics
import python
__all__ = ['ModelComparison']

from plots.plotter import PlotterParameters


class ModelComparison(object):

    def __init__(self, model1: NeuralModel, model2: NeuralModel) -> None:

        self.model1 = model1
        self.model2 = model2

    def compare(self,  neural_training: NeuralTraining,  train_loader: DataLoader, val_loader: DataLoader):
        logging.info(f'\nStart training {self.model1.model_id}')
        self.model1.train_model(training=neural_training, train_loader=train_loader, val_loader=val_loader)
        logging.info(f'\nStart training {self.model2.model_id}')

        # Reset/empty the dictionary of collected metrics
        neural_training.performance_metrics.reset()
        self.model2.train_model(training=neural_training, train_loader=train_loader, val_loader=val_loader)

    def load(self, plotter_parameters: PlotterParameters) -> None:
        from plots.plotter import Plotter
        perf_metric_model1 = PerformanceMetrics.load_summary(self.model1.model_id)
        perf_metric_model2 = PerformanceMetrics.load_summary(self.model2.model_id)
        logging.info(f'\n{perf_metric_model1=}\n{perf_metric_model2=}')

        # Merge plots
        merged_metrics = {k: [perf_metric_model1[k], perf_metric_model2[k]]
                          for k in perf_metric_model1.keys() & perf_metric_model2.keys()}
        for k, values in merged_metrics.items():
            setattr(plotter_parameters, 'y_label', k)
            setattr(plotter_parameters, 'title', f'{k} Comparison')
            Plotter.plot(values, [self.model1.model_id, self.model2.model_id], plotter_parameters)





