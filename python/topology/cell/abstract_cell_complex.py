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
from typing import Self, AnyStr, List, Tuple, Dict
# 3rd Party imports
import toponetx as tnx
import numpy as np
import torch
# Library imports
from topology.complex_element import ComplexElement
from topology.simplicial.simplicial_laplacian import SimplicialLaplacian
from topology.graph_complex_elements import GraphComplexElements
__all__ = ['AbstractCellComplex']


class AbstractCellComplex(object):
    def __init__(self, graph_complex_elements: GraphComplexElements) -> None:
        self.graph_complex_elements = graph_complex_elements

        edges_indices = [edge.node_indices for edge in graph_complex_elements.complex_edges]
        cell_indices = [edge.node_indices for edge in graph_complex_elements.complex_simplex_2]
        self.cell_indices = edges_indices + cell_indices

    def incidence_matrix(self, rank: int = 1, directed_graph: bool = True) -> np.array:
        """
        Extract the incidence matrix for a given rank and directed/undirected graph
        @param rank: Rank of the Simplicial complex
        @type rank: int
        @param directed_graph: Flag that specifies if the graph is directed
        @type directed_graph: bool
        @return: Incidence matrix
        @rtype: Numpy array
        """
        if rank < 0 or rank > 2:
            raise ValueError(f'Rank of incidence matrix {rank} should be [0, 2]')

        sc = tnx.CellComplex(self.cell_indices)
        _, _, incidence = sc.incidence_matrix(rank=rank, index=True, signed=directed_graph)
        return incidence.todense()

