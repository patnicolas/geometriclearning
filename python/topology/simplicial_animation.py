__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

from sympy.combinatorics import tetrahedron

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

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

# Example vertices
#                      0       1       2          3      4        5          6         7           8
vertices = np.array([[0, 0], [1, 0], [0.5, 1], [1, 1], [0, 1], [0, 0.5], [1, 1.5], [1.5, 1.5], [1.5, 0]])
edges = [(0, 1), (1, 2), (0, 2), (1, 3), (3, 4), (3, 7), (6, 7), (4, 6), (3, 5), (7, 8), (4, 5), (1, 8), (1, 7), [2, ]]
triangles = [[0, 1, 2], [4, 5, 3], [1, 8, 7]]
tetrahedrons = [[1, 8, 7, 3]]
colors = ['cyan', 'red', 'green', 'purple']

fig, ax = plt.subplots()
lines = []
patches = []

def init():
    ax.set_xlim(-0.1, 1.6)
    ax.set_ylim(-0.1, 1.7)
    plt.axis('off')
    return []

def update(frame):
    ax.clear()
    ax.set_xlim(-0.1, 1.6)
    ax.set_ylim(-0.1, 1.7)
    plt.axis('off')
    offset = 12
    start_edges = len(vertices) + offset
    start_triangles = start_edges + len(edges) + offset
    start_tetrahedrons = start_triangles + len(triangles) + offset
    title = ''
    if frame >= 0:
        title = 'Topological Set - Point Cloud'
        ax.scatter(vertices[:frame+1, 0], vertices[:frame+1, 1], s=300, c='blue')

    if frame >= start_edges:
        title = 'Undirected Graph'
        num_items = frame - start_edges
        if num_items >= len(edges):
            num_items = len(edges)
        for idx in range(num_items):
            i = edges[idx][0]
            j = edges[idx][1]
            ax.plot(*zip(vertices[i], vertices[j]), c='grey', linewidth=2)

    if frame >= start_triangles:
        title = 'Simplicial Complex'
        num_items = frame - start_triangles
        if num_items >= len(triangles):
            num_items = len(triangles)
        for idx in range(num_items):
            poly = Polygon(vertices[triangles[idx]], closed=True, alpha=0.2, color=colors[idx])
            ax.add_patch(poly)

    if frame >= start_tetrahedrons:
        title = 'Simplicial Complex'
        num_items = frame - start_tetrahedrons
        if num_items >= len(tetrahedrons):
            num_items = len(tetrahedrons)
        for idx in range(num_items):
            poly = Polygon(vertices[tetrahedrons[idx]], closed=True, alpha=0.5, color='darkgrey')
            ax.add_patch(poly)

    ax.set_title(title, fontdict={'fontsize': 22, 'fontname': 'Helvetica'})
    return []


ani = FuncAnimation(fig, update, frames=110, init_func=init, blit=False, interval=100, repeat=False)
plt.show()
