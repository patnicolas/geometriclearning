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
from typing import List, Dict, AnyStr, Any
# 3rd Party imports
import numpy as np
import matplotlib.pyplot as plt
# Library imports
from topology.simplicial_feature_set import SimplicialFeatureSet
__all__ = ['SimplicialVisualization']

class SimplicialVisualization(object):
    """
    Method dedicated to the visualization and animation of
    """
    triangle_colors = ['blue', 'red', 'green', 'purple', 'grey', 'orange', 'black']
    tetrahedron_color = 'lightgrey'

    def __init__(self, simplicial_feature_set: SimplicialFeatureSet, attributes: Dict[AnyStr, Any]):
        self.simplicial_feature_set = simplicial_feature_set
        self.attributes = attributes

    def show(self) -> None:
        """
        Display this simplicial domain with node, feature vectors, edges and faces.
        """
        import networkx as nx
        # Prepare the plots
        fig = plt.figure(figsize=(12, 10), facecolor='lightblue')
        fig.subplots()

        # Build the NetworkX
        G = nx.Graph()
        nodes = range(1, len(self.simplicial_feature_set.feature_set) + 1)
        G.add_nodes_from(nodes)
        G.add_edges_from(self.simplicial_feature_set.edge_set)

        for idx, feature in enumerate(self.simplicial_feature_set.feature_set):
            G.nodes[idx+1]['value'] = feature
        labels_map = self.__display_features()

        # Generate the x, y positions
        node_pos = nx.circular_layout(G, dim=2)
        face_label_pos = self.__face_label_pos(node_pos)

        # Draw the graph (nodes + edges)
        node_font_size = self.attributes.get('node_font_size', 16)
        nx.draw(G, node_pos, with_labels=True, node_color='cyan', node_size=380, font_size=node_font_size)
        label_pos = {idx: [v[0]-0.12, v[1]] for idx, v in node_pos.items()}
        bbox = dict(boxstyle="round,pad=0.2", edgecolor='black', facecolor='yellow')
        feature_font_size = self.attributes.get('feature_font_size', 10)
        nx.draw_networkx_labels(G,
                                label_pos,
                                labels=labels_map,
                                font_color='black',
                                font_size=feature_font_size,
                                bbox=bbox)

        # Draw the faces
        self.__draw_faces(node_pos=node_pos, face_label_pos=face_label_pos)
        plt.show()

    """ ------------------------------  Private Supporting Methods  ------------------------- """

    def __draw_faces(self, node_pos: np.array, face_label_pos: np.array) -> None:
        color_idx = 0
        label_offset = 0.15
        face_font_size = self.attributes.get('face_font_size', 15)

        # Draw the simplices (faces)
        tetrahedrons = []
        for idx, face in enumerate(self.simplicial_feature_set.face_set):
            face.append(face[0])
            face_pos = [node_pos[n] for n in face]
            x, y = zip(*face_pos)
            if len(face) == 5:
                tetrahedrons.append((x, y))
                """
                plt.text(x=face_label_pos[0][0],
                         y=face_label_pos[0][1] + label_offset,
                         s=f'Tetrahedron {idx + 1}',
                         fontdict={'fontsize': face_font_size, 'color': 'darkgrey'},
                         bbox=dict(boxstyle="round,pad=0.2", edgecolor='black', facecolor='white'))
                """

            else:
                face_color = SimplicialFeatureSet.triangle_colors[idx % len(SimplicialFeatureSet.triangle_colors)]
                plt.fill(x, y, face_color, alpha=0.2)
                plt.text(x=face_label_pos[color_idx][0] - label_offset,
                         y=face_label_pos[color_idx][1],
                         s=f'Triangle {idx + 1}',
                         fontdict={'fontsize': face_font_size-3, 'color': face_color},
                         bbox=dict(boxstyle="round,pad=0.2", edgecolor='black', facecolor='white'))
                color_idx += 1

        for (x, y) in tetrahedrons:
            plt.fill(x,
                     y,
                     color=SimplicialFeatureSet.tetrahedron_color,
                     alpha=0.6,
                     edgecolor='grey',
                     hatch='////')

    def __display_features(self) -> Dict[int, AnyStr]:
        def display_feature_values(x: np.array) -> AnyStr:
            return '\n'.join([f'{n:.2f}' for n in x])
        return {idx+1: display_feature_values(x) for idx, x in enumerate(self.simplicial_feature_set.feature_set)}

    def __face_label_pos(self, node_pos: Dict[int, np.array]) -> List[np.array]:
        def gravity_center(face_idx: List[int]) -> np.array:
            return (node_pos[face_idx[0]] + node_pos[face_idx[1]] + node_pos[face_idx[2]]) * 0.333 \
                if len(face_idx) == 3 \
                else (node_pos[face_idx[0]] + node_pos[face_idx[1]] + node_pos[face_idx[2]] + node_pos[face_idx[3]]) * 0.25

        face_labels_pos = [gravity_center(face_indices) for face_indices in self.simplicial_feature_set.face_set]
        return face_labels_pos

