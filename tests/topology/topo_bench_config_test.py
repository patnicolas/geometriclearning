import unittest

import logging
from json import JSONDecodeError

import python
from typing import Tuple, Dict
from omegaconf import DictConfig
from topology.topo_bench_config import TopoBenchConfig

class TBDataloaderTest(unittest.TestCase):

    def test_init(self):
        try:
            topo_bench_config = TopoBenchConfig(TBDataloaderTest.__create_config())
            loader_config = topo_bench_config('loader')
            logging.info(loader_config)
            self.assertTrue(True)
        except (KeyError, ValueError) as e:
            logging.error(e)
            self.assertTrue(False)

    def test_init2(self):
        try:
            topo_bench_config = TopoBenchConfig(TBDataloaderTest.__create_config())
            loader_config = topo_bench_config('loaderx')
            self.assertTrue(False)
        except (KeyError, ValueError) as e:
            self.assertTrue(True)

    def test_get_attributes(self):
        try:
            topo_bench_config = TopoBenchConfig(TBDataloaderTest.__create_config())
            split_config = topo_bench_config.get_split()
            logging.info(split_config)
            transform_config = topo_bench_config.get_transform()
            logging.info(transform_config)
            loss_config = topo_bench_config.get_tb_loss()
            logging.info(loss_config)
            evaluator_config = topo_bench_config.get_tb_evaluator()
            logging.info(evaluator_config)
            optimizer_config = topo_bench_config.get_tb_optimizer()
            logging.info(optimizer_config)
            self.assertTrue(True)
        except JSONDecodeError as e:
            logging.error(e)
            self.assertTrue(False)

    @staticmethod
    def __create_config() -> Tuple[Dict[str, DictConfig]]:
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
                    "k_value": 1, }
        }
        split_desc = {
            "learning_setting": "inductive",
            "split_type": "random",
            "data_seed": 0,
            "data_split_dir": "./data/MUTAG/splits/",
            "train_prop": 0.5,
        }
        readout_desc = {
            "readout_name": "PropagateSignalDown",
            "num_cell_dimensions": 1,
            "hidden_dim": 32,
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
            "metrics": ["accuracy", "precision", "recall"]
        }
        optimizer_desc = {
            "optimizer_id": "Adam",
            "parameters": {
                "lr": 0.001,
                "weight_decay": 0.0005}
        }
        return (
            loader_desc, transform_desc, split_desc, readout_desc, loss_desc, evaluator_desc, optimizer_desc
        )
