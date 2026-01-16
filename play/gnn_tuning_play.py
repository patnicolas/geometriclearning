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


# Python standard library imports
from typing import AnyStr
import logging
# 3rd Party imports
import optuna
from optuna.trial import TrialState
# Library imports
from play import Play
import python


class GNNTuningPlay(Play):
    """
    Source code related to the Substack article 'How to Tune a Graph Convolutional Network'.
    Article: https://patricknicolas.substack.com/p/how-to-tune-a-graph-convolutional
    Fisher-Rao:
        https://github.com/patnicolas/geometriclearning/blob/main/python/geometry/information_geometry/fisher_rao.py

    The features are implemented by the class GNNTuning in the source file
                  python/deeplearning/training/gnn_tuning.py
    The class GNNTuningPlay is a wrapper of the class GNNTuning
        The execution of the tests follows the same order as in the Substack article
    """
    def __init__(self, dataset_name: AnyStr, n_trials: int, timeout: int) -> None:
        super(GNNTuningPlay, self).__init__()
        self.dataset_name = dataset_name
        self.n_trials = n_trials
        self.timeout = timeout

    def play(self) -> None:
        """
        Implementation of the evaluation of GNN tuning in article 'How to Tune a Graph Convolutional Network' -
        Code snippet 6
        """
        from deeplearning.training.gnn_tuning import GNNTuning
        # We select the Tree-based Parzen Estimator for our HPO
        from optuna.samplers import TPESampler

        # We selected arbitrary Accuracy as our objective to maximize
        # The validation loss as our objective would require the direction 'minimize'
        study = optuna.create_study(study_name=self.dataset_name, sampler=TPESampler(), direction="maximize")
        study.optimize(GNNTuning.objective, n_trials=self.n_trials, timeout=self.timeout)

        # We select no deepcopy (=False) for memory efficient. There is no need
        # to preserve independent copies of each trial object.
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        logging.info(f'\nPruned trails:\n{pruned_trials}\nComplete trails:\n{complete_trials}')


if __name__ == "__main__":
    gnn_tuning_play = GNNTuningPlay(dataset_name='Flickr', n_trials=100, timeout=600)
    gnn_tuning_play.play()
