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


# Standard Library imports
import logging
from typing import Dict, Any
# 3rd Party imports
import toponetx as tnx
import networkx as nx
# Library imports
import python


""" 
Wrapping functions for lifting methods defined in TopoNetX
'Topological Lifting of Graph Neural Networks'
"""

def lift_from_graph_cliques(graph: nx.Graph, params: Dict[str, Any]) -> tnx.SimplicialComplex:
    from toponetx.transform import graph_to_clique_complex

    max_rank = params.get('max_rank', 2)
    logging.info(f'\nGraph lifted from NetworkX cliques with max rank {max_rank}')
    return graph_to_clique_complex(graph, max_rank=max_rank)

def lift_from_graph_neighbors(graph: nx.Graph, params: Dict[str, Any]) -> tnx.SimplicialComplex:
    from toponetx.transform import graph_to_neighbor_complex

    logging.info('Graph lifted from node neighbors')
    return graph_to_neighbor_complex(graph)

