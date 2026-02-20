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

# Standard library imports
from typing import Self, List
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

    def __init__(self, models: List[NeuralModel]) -> None:
        self.models = models

    @classmethod
    def build(cls, model1: NeuralModel, model2: NeuralModel) -> Self:
        return cls([model1, model2])

    def compare(self,  neural_training: NeuralTraining,  train_loader: DataLoader, val_loader: DataLoader):
        for idx, model in enumerate(self.models):
            logging.info(f'\nStart training {model.model_id}')
            # Reset/empty the dictionary of collected metrics for subsequent models
            if idx > 0:
                neural_training.performance_metrics.reset()
            model.train_model(training=neural_training, train_loader=train_loader, val_loader=val_loader)

    def load_and_plot(self, plotter_parameters: PlotterParameters) -> None:
        from plots.plotter import Plotter
        # Load the data from local files
        perf_metric_models = [PerformanceMetrics.load_summary(model.model_id) for model in self.models]

        # Merge performance metrics across multiple models
        merged_metrics = {k: [perf_metric_model[k] for perf_metric_model in perf_metric_models]
                          for k in perf_metric_models[0].keys()}

        # Setup plotting parameters then display the plot
        for k, values in merged_metrics.items():
            setattr(plotter_parameters, 'y_label', k)
            setattr(plotter_parameters, 'title', f'{k} Comparison')
            Plotter.plot(values, [model.model_id for model in self.models], plotter_parameters)

        Plotter.ioff()





