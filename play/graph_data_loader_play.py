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
from typing import AnyStr, Any, Dict
import logging
# Library imports
from play import Play
from dataset import DatasetException
from dataset.graph.graph_data_loader import GraphDataLoader
import python


class GraphDataLoaderPlay(Play):
    """
    Source code related to the Substack article 'Demystifying Graph Sampling & Walk Methods'.
    Reference: https://patricknicolas.substack.com/p/demystifying-graph-sampling-and-walk

    Source code for graph data loader
    https://github.com/patnicolas/geometriclearning/blob/main/python/dataset/graph/graph_data_loader.py

    The features are implemented by the class GraphDataLoader in the source file
                    python/dataset/graph/graph_data_loader.py
    The class GraphDataLoaderPlay is a wrapper of the class GraphDataLoader
    """
    def __init__(self,
                 dataset_name: AnyStr,
                 sampling_attributes: Dict[AnyStr, Any],
                 num_subgraph_nodes: int) -> None:
        super(GraphDataLoaderPlay, self).__init__()
        self.graph_data_loader = GraphDataLoader(dataset_name, sampling_attributes, num_subgraph_nodes)



    def play(self) -> None:
        """
        Source code related to Substack article 'Demystifying Graph Sampling & Walk Methods' - Code snippet 7
        Ref: https://patricknicolas.substack.com/p/demystifying-graph-sampling-and-walk
        """
        # Extract the loader for training and validation sets
        train_data_loader, test_data_loader = self.graph_data_loader()
        result = [f'{idx}: {str(batch)}'
                  for idx, batch in enumerate(train_data_loader) if idx < 3]
        logging.info('\nTrain data')
        logging.info('\n'.join(result))

if __name__ == "__main__":
    try:
        # Test 1 -------
        sampling_attrs = {
            'id': 'RandomNodeLoader',
            'num_parts': 256,
            'batch_size': 64,
            'num_workers': 2
        }
        dataset = 'Flickr'
        n_subgraph_nodes = 1024

        graph_data_loader_play = GraphDataLoaderPlay(dataset, sampling_attrs, n_subgraph_nodes)
        graph_data_loader_play.play()

        # Test 2 ------------
        sampling_attrs = {
            'id': 'NeighborLoader',
            'num_neighbors': [6, 2],
            'replace': True,
            'batch_size': 1024,
            'num_workers': 4
        }
        graph_data_loader_play = GraphDataLoaderPlay(dataset, sampling_attrs, n_subgraph_nodes)
        graph_data_loader_play.play()

        # Test 3 ------------
        sampling_attrs = {
            'id': 'GraphSAINTRandomWalkSampler',
            'walk_length': 3,
            'num_steps': 12,
            'sample_coverage': 100,
            'batch_size': 4096,
            'num_workers': 4
        }
        graph_data_loader_play = GraphDataLoaderPlay(dataset, sampling_attrs, n_subgraph_nodes)
        graph_data_loader_play.play()
    except (AssertionError, ValueError, DatasetException) as e:
        logging.error(e)


