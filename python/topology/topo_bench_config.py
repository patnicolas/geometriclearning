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
from typing import Dict, AnyStr, Any, Tuple, List
# 3rd Party library imports
from omegaconf import OmegaConf, DictConfig
from topobench.evaluator.evaluator import TBEvaluator
from topobench.loss.loss import TBLoss
from topobench.nn.readouts.propagate_signal_down import PropagateSignalDown
from topobench.optimizer import TBOptimizer

class TopoBenchConfig(object):
    config_keywords = {
        'data_domain': 'loader',
        'transform_type': 'transform',
        'learning_setting': 'split',
        'readout_name': 'readout',
        'dataset_loss': 'loss',
        'task': 'evaluator',
        'optimizer_id': 'optimizer'
    }

    def __init__(self, descriptors: Tuple[Dict[AnyStr, DictConfig]]) -> None:
        self._configs = {config_type: config for d in descriptors
                         for config_type, config in [TopoBenchConfig.__select_descriptor(d)]}

    def get_tb_evaluator(self) -> TBEvaluator:
        return TBEvaluator(**self._configs['evaluator'])

    def get_tb_optimizer(self) -> TBOptimizer:
        return TBOptimizer(**self._configs['optimizer'])

    def get_tb_propagate_signal_down(self) -> PropagateSignalDown:
        return PropagateSignalDown(**self._configs['readout'])

    def get_tb_loss(self) -> TBLoss:
        return TBLoss(**self._configs['loss'])

    def get_transform(self) -> DictConfig:
        return self._configs['transform']

    def get_loader(self) -> DictConfig:
        return self._configs['loader']

    def get_split(self) -> DictConfig:
        return self._configs['split']

    def __call__(self, config_name: AnyStr) -> DictConfig:
        return self._configs[config_name]

    """  --------------------  Private Supporting Methods -------------------------"""
    @staticmethod
    def __select_descriptor(descriptor: Dict[AnyStr, Any]) -> Tuple[AnyStr, DictConfig]:
        import json
        for keyword, config_type in TopoBenchConfig.config_keywords.items():
            if keyword in json.loads(f'" + {descriptor} + "'):
                return config_type, OmegaConf.create(descriptor)
        raise KeyError(f'No match for the descriptor {descriptor}')
