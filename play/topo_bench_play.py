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

from typing import Dict, AnyStr, Any, Tuple

from omegaconf import DictConfig
from topobench.data.loaders.graph.tu_datasets import TUDatasetLoader

# Library imports
from play import Play

import lightning as pl
# Hydra related imports
# Data related imports
from topobench.dataloader.dataloader import TBDataloader
from topobench.data.preprocessor import PreProcessor
# Model related imports
from topobench.model.model import TBModel
from topobench.nn.encoders.all_cell_encoder import AllCellFeatureEncoder


import torch
import torch.nn as nn
from topology.topo_bench_config import TopoBenchConfig


class TorchModel(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        #self.dim_hidden = hidden_size
        #self.in_channels = in_channels
        self.linear_0 = torch.nn.Linear(in_channels, hidden_size)
        self.linear_1 = torch.nn.Linear(in_channels, hidden_size)

    def forward(self, batch) -> Dict[AnyStr, Any]:
        x_0 = batch.x
        incidence_hyperedges = batch.incidence_hyperedges
        x_1 = torch.sparse.mm(incidence_hyperedges, x_0)

        x_0 = self.linear_0(x_0)
        x_0 = torch.relu(x_0)
        x_1 = self.linear_1(x_1)
        x_1 = torch.relu(x_1)
        model_out = {"labels": batch.y, "batch_0": batch.batch_0, "x_0": x_0, "hyperedge": x_1}
        return model_out


class TopoBenchWrapper(object):
    def __init__(self, descriptors: Tuple[Dict[str, DictConfig]], model: nn.Module) -> None:
        self.topo_bench_config = TopoBenchConfig(descriptors)
        self.data_loader = TopoBenchWrapper.__load_data_set(self.topo_bench_config)
        self.model = model

    def train(self, max_epochs: int, device_name: AnyStr = "cpu") -> None:
        model = self.__get_lightning_model()
        trainer = pl.Trainer(max_epochs=max_epochs,
                             min_epochs=2,
                             accelerator=device_name,
                             precision=32,
                             enable_progress_bar=True,
                             log_every_n_steps=1,
                             num_nodes=1)
        trainer.fit(model, self.data_loader)
        train_metrics = trainer.callback_metrics

        print('      Training metrics\n', '-' * 26)
        for key in train_metrics:
            print('{:<21s} {:>5.4f}'.format(key + ':', train_metrics[key].item()))

        trainer.test(model, self.data_loader)
        test_metrics = trainer.callback_metrics
        print(test_metrics)

    """ -----------------------  Private Supporting Methods -------------------- """

    def __get_lightning_model(self) -> TBModel:
        class LightningAdapter(pl.LightningModule):
            def __init__(self, model: nn.Module):
                super().__init__()
                self.model = model

            def forward(self, batch):
                return self.model(batch)

        return TBModel(backbone=LightningAdapter(self.model),
                       readout=self.topo_bench_config.get_tb_propagate_signal_down(),
                       loss=self.topo_bench_config.get_tb_loss(),
                       feature_encoder=AllCellFeatureEncoder(in_channels=[self.model.linear_0.in_features],
                                                             out_channels=out_channels),
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


if __name__ == '__main__':
    loader_desc = {
        "data_domain": "graph",
        "data_type": "TUDataset",
        "data_name": "MUTAG",
        "data_dir": "./data/MUTAG/"
    }

    transform_desc = {
        "khop_lifting":
        {
            "transform_type": "lifting",
            "transform_name": "HypergraphKHopLifting",
            "k_value": 1
        }
    }

    split_desc = {
        "learning_setting": "inductive",
        "split_type": "random",
        "data_seed": 0,
        "data_split_dir": "./data/MUTAG/splits/",
        "train_prop": 0.5,
    }

    # in_channels = 7
    out_channels = 2
    dim_hidden = 16

    readout_desc = {
        "readout_name": "PropagateSignalDown",
        "num_cell_dimensions": 1,
        "hidden_dim": dim_hidden,
        "out_channels": out_channels,
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
        "num_classes": out_channels,
        "metrics": ["accuracy", "precision", "recall"]
    }

    optimizer_desc = {
        "optimizer_id": "Adam",
        "parameters": {
            "lr": 0.001,
            "weight_decay": 0.0005}
        }

    my_descriptors: Tuple[Dict[str, DictConfig]] = (
        loader_desc, transform_desc, split_desc, readout_desc, loss_desc, evaluator_desc, optimizer_desc
    )

    topo_bench_wrapper = TopoBenchWrapper(descriptors=my_descriptors, model=TorchModel(hidden_size=16, in_channels=7))
    topo_bench_wrapper.train(max_epochs=3)


