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
__all__ = ['TopologicalDomain']




class TopologicalDomain(object):
    def __init__(self, graph_complex_elements: GraphComplexElements) -> None:
        """
        Constructor for the Simplicial Complex Model. Shape of Numpy array for the edge and face sets
        are evaluated for consistency.

        @param graph_complex_elements: Graph Complex elements for Node, Edges and Faces or cells
        @type graph_complex_elements:  GraphComplexElements
        """
        # Validate the shape of indices of the simplicial complex

        self.graph_complex_elements = graph_complex_elements
        # Extract the indices for the edges and faces
        edges_indices = [edge.node_indices for edge in graph_complex_elements.complex_edges]
        simplex_2_indices = [edge.node_indices for edge in graph_complex_elements.complex_simplex_2]
        self.simplicial_indices = edges_indices + simplex_2_indices

    def adjacency_matrix(self, directed_graph: bool = False) -> np.array:
        """
        Computation of the adjacency matrix (edges - nodes)
        @param directed_graph: Flag that specify if the graph is directed or not (Default Undirected graph)
        @type directed_graph: bool
        @return: Adjacency matrix as a dense matrix
        @rtype: Numpy array
        """
        # Initialize adjacency matrix
        n = len(np.concatenate([node.feature_set for node in self.graph_complex_elements.complex_nodes]))
        A = np.zeros((n, n), dtype=int)

        # Fill in edges
        for u, v in [edge.node_indices for edge in self.graph_complex_elements.complex_edges]:
            A[u-1, v-1] = 1
            if directed_graph:
                A[v-1, u-1] = 1
        return A