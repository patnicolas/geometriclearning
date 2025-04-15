__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from dl.training.neural_training import NeuralTraining
from dl.training.hyper_params import HyperParams
from dl.training.training_summary import TrainingSummary
from dl import GNNException
from plots.plotter import PlotterParameters
from metric.metric import Metric
from dl.training.exec_config import ExecConfig
from typing import Dict, AnyStr, Optional, List
import torch.nn as nn
import torch
import torch_geometric
from torch.utils.data import DataLoader
from tqdm import tqdm


class GNNTraining(NeuralTraining):

    def __init__(self,
                 hyper_params: HyperParams,
                 training_summary: TrainingSummary,
                 metrics: Dict[AnyStr, Metric],
                 exec_config: ExecConfig = ExecConfig.default(),
                 plot_parameters: Optional[List[PlotterParameters]] = None):
        """
        Default constructor for this variational auto-encoder
        @param hyper_params:  Hyper-parameters for training and optimizatoin
        @type hyper_params: HyperParams
        @param training_summary: Training monitoring
        @type training_summary: TrainingSummary
        @param metrics: Dictionary of metrics and values
        @type metrics: Dictionary
        @param exec_config: Configuration for optimization of execution of training
        @type exec_config: ExecConfig
        @param plot_parameters: Optional plotting parameters
        @type plot_parameters: List[PlotterParameters]
        """
        super(GNNTraining, self).__init__(hyper_params,
                                          training_summary,
                                          metrics,
                                          exec_config,
                                          plot_parameters)

    def __repr__(self) -> str:
        return repr(self.hyper_params)

    def train(self,
              model_id: AnyStr,
              neural_model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              val_enabled: bool = True) -> None:
        """
        Train and evaluation of a neural network given a data loader for a training set, a
        data loader for the evaluation/test1 set and an encoder_model. The weights of the various linear modules
        (neural_blocks) will be initialized if self.hyper_params using a Normal distribution

        @param model_id: Identifier for the model
        @type model_id: str
        @param neural_model: Neural model as torch module
        @type neural_model: nn_Module
        @param train_loader:  Data loader for the training set
        @type train_loader: torch_geometric.loader.DataLoader
        @param val_loader: Data loader for the evaluation set
        @param val_loader: torch_geometric.loader.DataLoader
        @param val_enabled: Enable validation
        @param val_enabled: bool
        """
        torch.manual_seed(42)

        # Force a conversion to 32-bits
        neural_model = neural_model.float()
        output_file_name = f'{model_id}: {self.plot_parameters[0].title}' if self.plot_parameters is not None \
            else model_id

        # Train and evaluation process
        for epoch in range(self.hyper_params.epochs):
            # Set training mode and execute training
            train_loss = self.__train(neural_model, epoch, train_loader)

            # Set mode and execute evaluation
            if val_enabled:
                eval_metrics = self.__val(neural_model, epoch, val_loader)
                self.training_summary(epoch, train_loss, eval_metrics)
            self.exec_config.apply_monitor_memory()

        # Generate summary
        self.training_summary.summary(output_file_name)
        print(f"\nMPS usage profile for\n{str(self.exec_config)}\n{self.exec_config.accumulator}")

    """ -----------------------------  Private helper methods ------------------------------  """

    def __train(self, neural_model: nn.Module, epoch: int, train_loader: DataLoader) -> torch.Tensor:
        neural_model.train()
        total_loss = 0.0
        optimizer = self.hyper_params.optimizer(neural_model)
        num_records = len(train_loader)
        model = neural_model.to(self.target_device)

        for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            try:
                # Force a conversion to float 32 if necessary
                if data.x.dtype == torch.float64:
                    data.x = data.x.float()

                # Move data to the GPU
                data = data.to(self.target_device)
                predicted = model(data)  # Call forward - prediction
                raw_loss = self.hyper_params.loss_function(predicted[data.train_mask], data.y[data.train_mask])

                # Set back propagation
                raw_loss.backward(retain_graph=True)
                loss = raw_loss.item()
                total_loss += loss

                # Monitoring and caching for performance
                self.exec_config.apply_batch_optimization(idx, optimizer)
                idx += 1
            except (RuntimeError | AttributeError | ValueError | Exception) as e:
                raise GNNException(str(e))
        return torch.tensor(total_loss / num_records)

    def __val(self, model: nn.Module, epoch: int, eval_loader: DataLoader) -> Dict[AnyStr, torch.Tensor]:
        total_loss = 0
        model.eval()
        metric_collector = {}

        # No need for computing gradient for evaluation (NO back-propagation)
        with torch.no_grad():
            count = 0
            for data in tqdm(eval_loader):
                try:
                    # Force a conversion to float 32 if necessary
                    if data.x.dtype == torch.float64:
                        data.x = data.x.float()

                    data = data.to(self.target_device)
                    predicted = model(data)  # Call forward - prediction
                    raw_loss = self.hyper_params.loss_function(predicted[data.val_mask], data.y[data.val_mask])

                    # Transfer prediction and labels to CPU for computing metrics
                    np_predicted = predicted.cpu().numpy()
                    np_labels = data.y.cpu().numpy()

                    # Update the metrics
                    for key, metric in self.metrics.items():
                        value = metric(np_predicted, np_labels)
                        metric_collector[key] = value

                    # Compute and accumulate the loss
                    total_loss += raw_loss.item()
                    count += 1
                except (RuntimeError | AttributeError| ValueError| Exception) as e:
                    raise GNNException(str(e))

        eval_loss = total_loss / count
        metric_collector[Metric.eval_loss_label] = torch.tensor(eval_loss)
        return metric_collector
