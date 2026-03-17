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

# Python library imports
from typing import Dict, AnyStr, Any, Tuple, List, Self, Literal, Callable
# 3rd Party library imports
import torch
import torch.nn as nn
import numpy as np
import lightning as pl

from omegaconf import DictConfig
from topobench.dataloader.dataloader import TBDataloader
from topobench.data.preprocessor import PreProcessor
from topobench.model.model import TBModel
from topobench.nn.encoders.all_cell_encoder import AllCellFeatureEncoder
from topobench.data.loaders.graph.tu_datasets import TUDatasetLoader
# Library imports
from metric.metric_type import MetricType
from topology.topo_bench_config import TopoBenchConfig


class TopoBenchWrapper(object):
    """
    Wrapper for automation and componentization of TopoBench functionality.
    TopoBench provides a unified benchmarking infrastructure for Topological Deep Learning (TDL) and Topological
    Data Analysis (TDA) by integrating and expanding upon current software tools.
    It combines NetworkX for graph processing with the TopoX suite—specifically TopoNetX for building complex
    structures and TopoModelX for model implementation. Additionally, it supports PyG models and original research code,
    offering a highly flexible environment for evaluating TDL.

    This wrapper is built from
    - Graph Model (MLP for demo purpose)
    - TopoBench JSON-formatted descriptors for loader, split, evaluator, dataset, optimizer ...
        Examples:
             transform_desc = {
                "khop_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "HypergraphKHopLifting",
                    "k_value": k
                }
            }
            loss_desc = {
                "dataset_loss": {
                    "task": "classification",
                    "loss_type": "cross_entropy"
                }
            }
    Notes:
        - TopoBench library relies on Torch Lightning modules so any model inheriting from nn.Module has to be converted
          to a LightningModule though a simple Adapter pattern
        - The method 'build' is an alternative constructor and an example of a parameterized the TopoBench
          configuration to make evaluation more flexible.

    References:
        Paper: "TopoBench: A Framework for Benchmarking Topological Deep Learning" (https://arxiv.org/pdf/2406.06642)
        GitHub: https://github.com/geometric-intelligence/TopoBench
    """
    def __init__(self, graph_network: nn.Module, topo_bench_descriptors: Tuple[Dict[str, DictConfig]]) -> None:
        """
        Constructor for the wrapper around TopoBench library
        @param graph_network: Multi-perceptron model to process graph
        @type graph_network: Inherited from nn.Module
        @param topo_bench_descriptors: JSON formatted descriptors for TopoBench configuration
        @type topo_bench_descriptors: Tuple[Dict[str, DictConfig]]
        """
        self.graph_network = graph_network
        self.topo_bench_config = TopoBenchConfig(topo_bench_descriptors)
        self.data_loader = TopoBenchWrapper.__load_data_set(self.topo_bench_config)
        self.collected_metrics: Dict[MetricType, List[np.array]] = {}

    def __str__(self) -> AnyStr:
        data_loader_desc = (f'Name: {self.data_loader.name},  Batch size: {self.data_loader.batch_size}, '
                            f'{len(self.data_loader.dataset_train)} Training samples',
                            f'{len(self.data_loader.dataset_test)} Test samples')
        return (f'\nGraph Network\n{self.graph_network}\nData loader:{data_loader_desc}\nCollected Metrics'
                f'\n{self.collected_metrics}')

    @classmethod
    def build(cls, graph_network: nn.Module, k_value: int, data_name: AnyStr, lr: float) -> Self:
        """
        Alternative constructor for a predefined TopoBench configuration using a predefined, parameterized descriptors.
        This implementation is an example of parameterization of the configuration for TopoBench using k_value for
        k-hop lifting, data set name and the learning rate used by the optimizer

        @param graph_network: Graph Neural Network
        @type graph_network:
        @param k-value: K-value for the k-hop lifting
        @type k-value: int
        @param data_name: Name of the data set
        @type data_name: str
        @param lr: Learning rate for optimizer used in training
        @type lr: float
        @return: Instance of TopoBenchWrapper
        @rtype: TopoBenchWrapper
        """
        if lr < 1e-6 or lr > 0.05:
            raise ValueError(f'Learning rate {lr} should be a value between 1e-6 and 0.05')
        if k_value < 1 or k_value > 8:
            raise ValueError(f'k_value {k_value} should be a number between 1 and 8')

        # Retrieve the size of the hidden layer from the Torch model
        dim_hidden = graph_network.get_dim_hidden()
        # Loads the pre-defined configuration descriptor
        descriptors = TopoBenchWrapper.__get_config_descriptors(dim_hidden, k_value, data_name, lr)
        return cls(graph_network, descriptors)

    def train(self,
              max_epochs: int,
              float_precision: Literal[64, 32, 16],
              device_name: Literal['auto', 'cpu', 'cuda'] = "cpu") -> None:
        """
        Trigger training & evaluation using PyTorch Lightning library. TopoBench relies on Torch Lightning library,
        and therefore we need to convert our model to a Lightning module.

        @param max_epochs: Maximum number of epochs used in training
        @type max_epochs: int
        @param float_precision: Floating point precision
        @type float_precision: Literal
        @param device_name: Supported devices 'auto, 'cpuy', 'cuda' ...
        @type device_name: str
        """
        if max_epochs < 2 or max_epochs > 512:
            raise ValueError(f'max_epochs {max_epochs} should be a number between 2 and 512')

        # Convert the torch module into a torch lightning module
        lightning_graph_model = self.__get_lightning_graph_model()
        # Initialize the lightning trainer
        trainer = pl.Trainer(max_epochs=max_epochs,
                             min_epochs=1,
                             accelerator=device_name,
                             precision=float_precision,
                             enable_progress_bar=False,
                             log_every_n_steps=1,
                             num_nodes=1)
        # Train the lightning model
        trainer.fit(lightning_graph_model, self.data_loader)
        # Collect the metric related to training
        train_metrics = trainer.callback_metrics

        # Test and collect metrics related to the test
        trainer.test(lightning_graph_model, self.data_loader)
        test_metrics = trainer.callback_metrics

        # Prepare metrics for plotting
        self.__update_metrics(train_metrics, test_metrics)

    """ -----------------------  Private Supporting Methods -------------------- """

    def __update_metrics(self,
                         train_metrics: Dict[AnyStr, torch.Tensor],
                         test_metrics: Dict[AnyStr, torch.Tensor]) -> None:
        train_metrics_map = {
            'train/loss': MetricType.TrainLoss
        }
        test_metrics_map = {
            'test/accuracy': MetricType.Accuracy,
            'test/precision': MetricType.Precision,
            'test/recall': MetricType.Recall,
            'test/loss': MetricType.EvalLoss,
            'test/f1': MetricType.F1
        }
        self.__collect_metrics(train_metrics_map, train_metrics)
        self.__collect_metrics(test_metrics_map, test_metrics)

    def __collect_metrics(self, metrics_map: Dict[AnyStr, MetricType], metrics: Dict[AnyStr, torch.Tensor]) -> None:
        for key, metric in metrics_map.items():
            if metric in self.collected_metrics:
                self.collected_metrics[metric].append(metrics[key].numpy())
            else:
                self.collected_metrics[metric] = [metrics[key].numpy()]

    def __get_lightning_graph_model(self) -> TBModel:
        class LightningModelAdapter(pl.LightningModule):
            def __init__(self, model: nn.Module):
                super().__init__()
                self.model = model

            def forward(self, batch):
                return self.model(batch)

        all_in_channels = [self.graph_network.get_in_channels()]
        return TBModel(backbone=LightningModelAdapter(self.graph_network),
                       readout=self.topo_bench_config.get_tb_propagate_signal_down(),
                       loss=self.topo_bench_config.get_tb_loss(),
                       feature_encoder=AllCellFeatureEncoder(in_channels=all_in_channels,
                                                             out_channels=self.graph_network.out_channels),
                       evaluator=self.topo_bench_config.get_tb_evaluator(),
                       optimizer=self.topo_bench_config.get_tb_optimizer(),
                       compile=False)

    @staticmethod
    def __load_data_set(topo_bench_config: TopoBenchConfig) -> TBDataloader:
        graph_loader = TUDatasetLoader(topo_bench_config.get_loader())

        dataset, dataset_dir = graph_loader.load()
        preprocessor = PreProcessor(dataset, dataset_dir, topo_bench_config.get_transform())
        dataset_train, dataset_val, dataset_test = preprocessor.load_dataset_splits(topo_bench_config.get_split())
        return TBDataloader(dataset_train, dataset_val, dataset_test, batch_size=32)

    @staticmethod
    def __get_config_descriptors(dim_hidden: int, k: int, data_name: AnyStr, lr: float) -> Tuple[Dict[str, DictConfig]]:
        """
        This is an example of a pre-defined, parameterized parameters for configuration of TopoBench
        """
        loader_desc = {
            "data_domain": "graph",
            "data_type": "TUDataset",
            "data_name": data_name,
            "data_dir": f"./data/{data_name}/"
        }
        transform_desc = {
            "khop_lifting": {
                "transform_type": "lifting",
                "transform_name": "HypergraphKHopLifting",
                "k_value": k
            }
        }
        split_desc = {
            "learning_setting": "inductive",
            "split_type": "random",
            "data_seed": 0,
            "data_split_dir": f"./data/{data_name}/splits/",
            "train_prop": 0.5,
        }
        readout_desc = {
            "readout_name": "PropagateSignalDown",
            "num_cell_dimensions": 1,
            "hidden_dim": dim_hidden,
            "out_channels": 2,
            "task_level": "graph",
            "pooling_type": "sum",
        }
        loss_desc = {
            "dataset_loss": {
                "task": "classification",
                "loss_type": "cross_entropy"
            }
        }
        evaluator_desc = {
            "task": "classification",
            "num_classes": 2,
            "metrics": ["accuracy", "precision", "recall", "f1"]
        }
        optimizer_desc = {
            "optimizer_id": "Adam",
            "parameters": {
                "lr": lr,
                "weight_decay": 0.0012
            }
        }
        return (
            loader_desc, transform_desc, split_desc, readout_desc, loss_desc, evaluator_desc, optimizer_desc
        )



