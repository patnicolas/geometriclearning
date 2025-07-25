_author__ = "Patrick Nicolas"
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

from typing import Self, AnyStr, List, Dict, Tuple
import toponetx as tnx
import matplotlib.pyplot as plt
import numpy as np
import torch
from enum import Enum
from dataclasses import dataclass
from topology import TopologyException

class SimplicialLaplacianType(Enum):
    UpLaplacian = 'Upper-Laplacian'
    DownLaplacian = 'Lower-Laplacian'
    HodgeLaplacian = 'Hodge-Laplacian'


@dataclass
class SimplicialLaplacian:
    """
    Define the components of the Laplacian for Simplicial Complexes
    @param simplicial_laplacian_type Type of Laplacian (UP, DOWN or Hodge)
    @param rank Rank of the Laplacian
    @param signed Boolean flag to specify if the values of Laplacian are signed (Directed/Undirected)
    """
    simplicial_laplacian_type:  SimplicialLaplacianType
    rank: int
    signed: bool

    def __str__(self) -> AnyStr:
        return f'{self.simplicial_laplacian_type.value}, rank={self.rank}, signed={self.signed}'

    def __call__(self, simplicial_indices: List[List[int]]) -> np.array:
        """
        Compute the various combination of Laplacian (UP, DOWN, Hodge) for different rank.

        @param simplicial_indices: List of edge and face indices
        @type simplicial_indices: List of list
        @return: 2D Numpy array representing the Laplacian matrix
        @rtype: Numpy array
        """
        if len(simplicial_indices) < 1:
            raise TopologyException('Cannot compute simplicial Laplacian with undefined indices')
        try:
            sc = tnx.SimplicialComplex(simplicial_indices)
            match self.simplicial_laplacian_type:
                case SimplicialLaplacianType.UpLaplacian:
                    laplacian_matrix = sc.up_laplacian_matrix(self.rank, self.signed)
                case SimplicialLaplacianType.DownLaplacian:
                    laplacian_matrix = sc.down_laplacian_matrix(self.rank, self.signed)
                case SimplicialLaplacianType.HodgeLaplacian:
                    laplacian_matrix = sc.hodge_laplacian_matrix(self.rank, self.signed)
            return laplacian_matrix.toarray()
        except ValueError as e:
            raise TopologyException(e)


class SimplicialFeatureSet(object):
    """
    Implementation of the Simplicial Complex with operators and a feature set (embedded vector).
    The functionality is:
    - Computation of incidence and adjacency matrices
    - Computation of various Laplacian operators
    - Visualization of Simplicial Complexes
    """
    triangle_colors = ['blue', 'red', 'green', 'purple', 'grey', 'orange']
    tetrahedron_color = 'lightgrey'

    def __init__(self, feature_set: np.array, edge_set: List[List[int]], face_set: List[List[int]]) -> None:
        """
        Constructor for the Simplicial Complex Model. Shape of Numpy array for the edge and face sets
        are evaluated for consistency.
        
        @param feature_set: Feature set or feature vector 
        @type feature_set: Numpy array
        @param edge_set:  Edge set an array of pair of node indices
        @type edge_set: Numpy array
        @param face_set:  Face set as an array of 3 node indices
        @type face_set: Numpy array
        """
        # Validate the shape of indices of the simplicial complex
        SimplicialFeatureSet.__validate(edge_set, face_set)
        
        self.feature_set = feature_set
        # Tuple (Src -> Dest)
        self.edge_set = edge_set
        # Either triangle {x, y, z] or Tetrahedron {x, y, z, t}
        self.face_set = face_set
        self.simplicial_indices = self.edge_set + self.face_set

    @classmethod
    def build(cls, dimension: int, edge_set: List[List[int]], face_set: List[List[int]]) -> Self:
        """
        Alternative constructor for the Simplicial model that uses random value for features set. The size of the
        feature set matrix is computed from the list of edges node indices.
        The feature set is the matrix number of nodes x dimension as follows:
                Feature#1   Feature#2  ...  Feature#dimension
        Node 1
        Node 2

        @param dimension: Size of the feature vectors
        @type dimension: int
        @param edge_set: Edge set as a tensor of pair of node indices
        @type edge_set: Torch tensor
        @param face_set:  Face set as a tensor of tensor with 3 node indices
        @type face_set: Torch tensor
        @return: Instance of Simplicial model
        @rtype: SimplicialFeatureSet
        """
        import itertools
        assert dimension > 0, f'Dimension of random vector {dimension} should be > 0'

        num_nodes = max(list(itertools.chain.from_iterable(edge_set)))
        random_feature_set = torch.rand(num_nodes, dimension)
        return cls(random_feature_set, edge_set, face_set)

    def __str__(self) -> AnyStr:
        return f'\nFeatures:\n{self.feature_set}\nEdges:\n{self.edge_set}\nFaces:\n{self.face_set}'

    def adjacency_matrix(self, directed_graph: bool = False) -> np.array:
        """
        Computation of the adjacency matrix (edges - nodes)
        @param directed_graph: Flag that specify if the graph is directed or not (Default Undirected graph)
        @type directed_graph: bool
        @return: Adjacency matrix as a dense matrix
        @rtype: Numpy array
        """
        # Initialize adjacency matrix
        n = len(self.feature_set)
        A = np.zeros((n, n), dtype=int)

        # Fill in edges
        for u, v in self.edge_set:
            A[u-1, v-1] = 1
            if directed_graph:
                A[v-1, u-1] = 1
        return A

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
        assert 0 <= rank < 3, f'Rank of incidence matrix {rank} should be [0, 2]'

        sc = tnx.SimplicialComplex(self.simplicial_indices)
        _, _, incidence = sc.incidence_matrix(rank=rank, index=True, signed=directed_graph)
        return incidence.todense()

    def laplacian(self, simplicial_laplacian: SimplicialLaplacian) -> np.array:
        return simplicial_laplacian(self.simplicial_indices)

    def show(self) -> None:
        """
        Display this simplicial domain with node, feature vectors, edges and faces.
        """
        import networkx as nx
        # Prepare the plots
        fig = plt.figure(figsize=(8, 6), facecolor='lightblue')
        fig.subplots()

        # Build the NetworkX
        G = nx.Graph()
        nodes = range(1, len(self.feature_set) + 1)
        G.add_nodes_from(nodes)
        G.add_edges_from(self.edge_set)

        for idx, feature in enumerate(self.feature_set):
            G.nodes[idx+1]['value'] = feature
        labels_map = self.__display_features()

        # Generate the x, y positions
        node_pos = nx.circular_layout(G, dim=2)
        face_label_pos = self.__face_label_pos(node_pos)

        # Draw the graph (nodes + edges)
        nx.draw(G, node_pos, with_labels=True, node_color='cyan', node_size=380, font_size=17)
        label_pos = {idx: [v[0]-0.12, v[1]] for idx, v in node_pos.items()}
        bbox = dict(boxstyle="round,pad=0.2", edgecolor='black', facecolor='yellow')
        nx.draw_networkx_labels(G, label_pos, labels=labels_map, font_color='black', font_size=10, bbox=bbox)

        # Draw the faces
        self.__draw_faces(node_pos=node_pos, face_label_pos=face_label_pos)
        plt.show()

    """ -------------------------  Private Supporting methods ------------------ """

    def __draw_faces(self, node_pos: np.array, face_label_pos: np.array) -> None:
        color_idx = 0
        label_offset = 0.15

        # Draw the simplices (faces)
        for idx, face in enumerate(self.face_set):
            face.append(face[0])
            face_pos = [node_pos[n] for n in face]
            x, y = zip(*face_pos)
            if len(face) == 5:
                plt.fill(x, y, color=SimplicialFeatureSet.tetrahedron_color, alpha=0.7)
                """
                    plt.text(x=face_label_pos[0][0],
                             y=face_label_pos[0][1] + label_offset,
                             s=f'Tetrahedron {idx + 1}',
                             fontdict={'fontsize': 16, 'color': 'darkgrey'},
                             bbox=dict(boxstyle="round,pad=0.2", edgecolor='black', facecolor='white'))
                """
            else:
                face_color = SimplicialFeatureSet.triangle_colors[idx % len(SimplicialFeatureSet.triangle_colors)]
                plt.fill(x, y, face_color, alpha=0.4)
                plt.text(x=face_label_pos[color_idx][0] - label_offset,
                         y=face_label_pos[color_idx][1],
                         s=f'Triangle {idx + 1}',
                         fontdict={'fontsize': 13, 'color': face_color},
                         bbox=dict(boxstyle="round,pad=0.2", edgecolor='black', facecolor='white'))
                color_idx += 1

    @staticmethod
    def __validate(edge_set: np.array, face_set: np.array) -> None:
        assert len(edge_set) > 0, 'Simplicial requires at least one edge'
        assert all(len(sublist) == 2 for sublist in edge_set), f'All elements of edge list should have 2 indices'

        assert len(face_set) > 0, 'Simplicial requires at least face'
        assert all(len(sublist) in (3, 4) for sublist in face_set), \
            f'All elements of edge list should have 3 or 4 indices'

    def __display_features(self) -> Dict[int, AnyStr]:
        def display_feature_values(x: np.array) -> AnyStr:
            return '\n'.join([f'{n:.2f}' for n in x])
        return {idx+1: display_feature_values(x) for idx, x in enumerate(self.feature_set)}

    def __face_label_pos(self, node_pos: Dict[int, np.array]) -> List[np.array]:
        def gravity_center(face_idx: List[int]) -> np.array:
            return (node_pos[face_idx[0]] + node_pos[face_idx[1]] + node_pos[face_idx[2]]) * 0.333 \
                if len(face_idx) == 3 \
                else (node_pos[face_idx[0]] + node_pos[face_idx[1]] + node_pos[face_idx[2]] + node_pos[face_idx[3]]) * 0.25

        face_labels_pos = [gravity_center(face_indices) for face_indices in self.face_set]
        return face_labels_pos
