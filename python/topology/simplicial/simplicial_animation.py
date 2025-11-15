__author__ = "Patrick Nicolas"
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
from typing import List, Tuple, Dict,AnyStr, Any
# 3rd Party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
__all__ = ['SimplicialAnimation']


class SimplicialAnimation(object):
    """
    Two steps:
    1. Represent simplices of different dimensions (vertices, edges, triangles, tetrahedra).
        0-simplices (vertices): Scatter points.
        1-simplices (edges): Lines between points.
        2-simplices (triangles): Use Polygon from matplotlib.patches.
        3-simplices (tetrahedra): Not native in 3D Matplotlib; use Poly3DCollection from mpl_toolkits.mplot3d.art3d.

    2. Show how the complex evolves—either over a filtration (like in persistent homology) or structural construction
    (growing the complex).

    Animate steps like:
    - Birth of a vertex (0-simplex)
    - Adding edges
    - Filling in triangles
    - Growing tetrahedra
    This mimics the process of building a Čech or Vietoris–Rips complex over increasing radius.

    Configuration dictionary
        fig_size: Tuple[int, int]
        x_lim: Tuple[float, float]   -0.1, 1.6
        y_lim: Tuple[float, float]     -0.1, 1.7
        group_interval: int   20
        logo_pos: Tuple[float, float]   -0.3, 1.85
        title_pos: Tuple[float, float]  0.6, 0.94
        status_pos: Tuple[float, float]  0.05  -0.15
        fps: int   ex 10
        interval: int  ex: 200
        num_frames: int
    """

    face_colors = ['cyan', 'red', 'green', 'purple', 'yellow', 'magenta', 'orange', 'blue', 'pink', 'navy', 'teal']
    tetrahedron_hatches = ['///', '--', '|||']
    tetrahedron_colors = ['blue', 'green', 'red']

    def __init__(self,
                 nodes: np.array,
                 edge_set: List[Tuple[int, int]],
                 face_set: List[List[int]],
                 config: Dict[AnyStr, Any]) -> None:
        if len(nodes) == 0:
            raise ValueError('Cannot animate a simplicial with no node')

        self.nodes = nodes
        self.edge_set = edge_set

        # Extract triangles and tetrahedrons
        self.triangle_set, self.tetrahedron_set = [], []
        [( self.triangle_set if len(face) == 3 else self.tetrahedron_set ).append(face) for face in face_set]

        fig_size = config.get('fig_size', (10, 9))
        self.config = config
        self.fig, self.ax = plt.subplots(figsize=fig_size)

    def show(self, save: bool = True) -> None:
        def set_axis() -> None:
            self.ax.set_xlim(self.config['xlim'])
            self.ax.set_ylim(self.config['ylim'])
            plt.axis('off')

        def init() -> List:
            set_axis()
            return []

        def update(frame: int) -> List:
            if frame < 125:
                self.ax.clear()
                set_axis()
                self.fig.set_facecolor('#f0f9ff')
                self.ax.set_facecolor('#f0f9ff')
                group_interval = self.config.get('group_interval', 18)
                start_edges = len(self.nodes) + group_interval
                start_triangles = start_edges + len(self.edge_set) + group_interval
                start_tetrahedrons = start_triangles + len(self.triangle_set) + group_interval
                title = ''
                status = []
                if frame >= 0:
                    title = 'Point Cloud | 0-simplices'
                    self.ax.scatter(self.nodes[:frame + 1, 0], self.nodes[:frame + 1, 1], s=300, c='blue')
                    num_nodes = frame if frame < len(self.nodes) else len(self.nodes)
                    status.append(f'{num_nodes} nodes')

                if frame >= start_edges:
                    title = 'Undirected Graph | 1-simplices'
                    num_items = SimplicialAnimation.__num_items(frame - start_edges, len(self.edge_set))
                    for idx in range(num_items):
                        i = self.edge_set[idx][0]
                        j = self.edge_set[idx][1]
                        self.ax.plot(*zip(self.nodes[i], self.nodes[j]), c='grey', linewidth=2)
                    status.append(f'{num_items} edges')

                if frame >= start_triangles:
                    title = 'Simplicial Complex | 2-simplices'
                    num_items = SimplicialAnimation.__num_items(frame - start_triangles, len(self.triangle_set))
                    for idx in range(num_items):
                        poly = Polygon(self.nodes[self.triangle_set[idx]],
                                       closed=True,
                                       alpha=0.2,
                                       color=SimplicialAnimation.__attribute(SimplicialAnimation.face_colors, idx))
                        self.ax.add_patch(poly)
                    status.append(f'{num_items} triangles')

                if frame >= start_tetrahedrons:
                    title = 'Simplicial Complex | 3-simplices'
                    num_items = SimplicialAnimation.__num_items(frame - start_tetrahedrons, len(self.tetrahedron_set))
                    for idx in range(num_items):
                        poly = Polygon(self.nodes[self.tetrahedron_set[idx]],
                                       closed=True,
                                       alpha=0.2,
                                       color=SimplicialAnimation.__attribute(SimplicialAnimation.tetrahedron_colors, idx),
                                       edgecolor='black',
                                       hatch=SimplicialAnimation.__attribute(SimplicialAnimation.tetrahedron_hatches, idx))
                        self.ax.add_patch(poly)
                    status.append(f'{num_items} tetrahedrons')

                self.__descriptors(status, title)
                return []

        interval = self.config.get('interval', 100)
        num_frames = self.config.get('num_frames', 100)
        ani = FuncAnimation(self.fig,
                            update,
                            frames=num_frames,
                            init_func=init,
                            blit=False,
                            interval=interval,
                            repeat=False)
        if save:
            fps = self.config.get('fps', 12)
            ani.save('../../animation/simplicial_anim.mp4', writer='ffmpeg', fps=fps, dpi=300)
        else:
            plt.show()

    """  -------------------------  Private Helper Methods ------------------------ """
    @staticmethod
    def __num_items(n_items: int, max_num_items: int) -> int:
        return n_items if n_items < max_num_items else max_num_items

    @staticmethod
    def __attribute(items: List[AnyStr], idx: int) -> AnyStr:
        return items[idx % len(items)]

    def __descriptors(self, status: List[AnyStr], title: AnyStr) -> None:
        self.ax.text(x=self.config['status_pos'][0],
                     y=self.config['status_pos'][1],
                     s=', '.join(status),
                     fontdict={'fontsize': 16, 'fontname': 'Helvetica', 'color': 'grey'})
        self.ax.set_title(x=self.config['title_pos'][0],
                          y=self.config['title_pos'][1],
                          label=title,
                          fontdict={'fontsize': 20, 'fontname': 'Helvetica', 'color': 'black'})
        self.ax.text(x=self.config['logo_pos'][0],
                     y=self.config['logo_pos'][1],
                     s='Hands-on Geometric Deep Learning',
                     fontdict={'fontsize': 18, 'fontname': 'Apple Chancery', 'color': 'blue'})


if __name__ == '__main__':
    configuration = {
        'fig_size': (8, 6),
        'xlim': (-0.1, 1.6),
        'ylim': (-0.1, 2.2),
        'group_interval': 20,
        'logo_pos': (-0.35, 2.38),
        'title_pos': (0.6, 0.95),
        'status_pos': (0.05, -0.18),
        'fps': 12,
        'interval': 90,
        'num_frames': 120
    }
    vertices = np.array([
        [0, 0], [1, 0], [0.5, 1], [1, 1], [0, 1], [0, 0.5], [1, 1.5], [1.5, 1.5], [1.5, 0], [0, 1.5], [1, 0.5],
        [0.5, 1.5], [0.5, 2.0], [0.0, 2.0], [1.5, 2.0]
    ])
    edges = [(0, 1), (1, 2), (0, 2), (1, 3), (3, 4), (3, 7), (6, 7), (4, 6), (2, 5), (7, 8), (4, 5), (1, 8), (1, 7),
             [2, 4], [3, 6], (2, 6), [4, 9], [9, 6], [2, 10], [4, 11], [11, 12],[12, 6], [13, 4], [14, 7]]
    faces = [[0, 1, 2], [4, 5, 2], [1, 8, 7], [3, 6, 7], [4, 9, 11], [4, 11, 6], [4, 6, 2], [2, 3, 10], [12, 11, 6],
             [1, 8, 7, 3], [5, 4, 6, 2], [4, 9, 11, 6]]

    simplicial_animation = SimplicialAnimation(nodes=vertices, edge_set=edges, face_set=faces, config=configuration)
    simplicial_animation.show()
