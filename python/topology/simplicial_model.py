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
import numpy as np
import torch


class SimplicialModel(object):

    def __init__(self, feature_set: np.array, edge_set: np.array, face_set: np.array) -> None:
        # Validate the shape of indices of the simplicial complex
        SimplicialModel.__validate(edge_set, face_set)
        self.feature_set = feature_set
        self.edge_set = edge_set.tolist()
        self.face_set = face_set.tolist()
        self.simplicial_indices = self.edge_set + self.face_set

    @classmethod
    def build(cls, feature_set: torch.Tensor, edge_set: np.array, face_set: np.array) -> Self:
        return cls(feature_set.numpy(), edge_set, face_set)

    def __str__(self) -> AnyStr:
        return f'\nFeatures:\n{self.feature_set}\nEdges:\n{self.edge_set}\nFaces:\n{self.face_set}'

    def adjacency_matrix(self, undirected: bool = True) -> np.array:
        # Initialize adjacency matrix
        n = len(self.feature_set)
        A = np.zeros((n, n), dtype=int)

        # Fill in edges
        for u, v in self.edge_set:
            A[u-1, v-1] = 1
            if undirected:
                A[v-1, u-1] = 1
        return A

    def incidence_matrix(self, rank: int = 1, directed_graph: bool = True) -> np.array:
        assert 0 <= rank < 3, f'Rank of incidence matrix {rank} should be [0, 2]'

        sc = tnx.SimplicialComplex(self.simplicial_indices)
        incidence_row, incidence_cols, incidence = sc.incidence_matrix(rank=rank, index=True, signed=directed_graph)
        return incidence.todense()

    def show(self) -> None:
        import networkx as nx
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 7))
        fig.patch.set_facecolor('#F2F9FE')
        colors = ['blue', 'red', 'green', 'cyan', 'white', 'orange', 'yellow']

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
        nx.draw(G, node_pos, with_labels=True, node_color='cyan', node_size=380, font_size=20)
        label_pos = {idx: [v[0]-0.12, v[1]] for idx, v in node_pos.items()}
        label_offset = 0.1
        bbox = dict(boxstyle="round,pad=0.2", edgecolor='black', facecolor='yellow')
        nx.draw_networkx_labels(G, label_pos, labels=labels_map, font_color='black', font_size=12, bbox=bbox)

        # Draw the simplices (faces)
        for idx, face in enumerate(self.face_set):
            face.append(face[0])
            face_pos = [node_pos[n] for n in face]
            x, y = zip(*face_pos)
            plt.fill(x, y, color=colors[idx % len(colors)], alpha=0.3)
            plt.text(x=face_label_pos[idx][0] - label_offset,
                     y=face_label_pos[idx][1],
                     s=f'Face {idx + 1}',
                     fontdict={'fontsize': 16})
        plt.show()

    @staticmethod
    def __validate(edge_set: np.array, face_set: np.array) -> None:
        assert edge_set.shape[1] == 2, f'Shape of edge set {edge_set.shape} should be (_, 2)'
        assert face_set.shape[1] == 3, f'Shape of edge set {face_set.shape} should be (_, 3)'

    def __display_features(self) -> Dict[int, AnyStr]:
        def display_feature_values(x: np.array) -> AnyStr:
            return '\n'.join([f'{n:.2f}' for n in x])
        return {idx+1: display_feature_values(x) for idx, x in enumerate(self.feature_set)}

    def __face_label_pos(self, node_pos: Dict[int, np.array]) -> List[np.array]:
        def gravity_center(face_indices: List[int]) -> np.array:
            center = (node_pos[face_indices[0]] + node_pos[face_indices[1]] + node_pos[face_indices[2]]) * 0.333
            return center

        face_labels_pos = [gravity_center(face_indices) for face_indices in self.face_set]
        return face_labels_pos
