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
vertices = np.array([[0, 0], [1, 0], [0.5, 1]])
edges = [(0, 1), (1, 2), (0, 2)]
triangle = [0, 1, 2]

fig, ax = plt.subplots()
points = ax.scatter([], [], c='blue')
lines = []
patches = []

def init():
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    return []

def update(frame):
    ax.clear()
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)

    if frame >= 0:
        ax.scatter(vertices[:frame+1, 0], vertices[:frame+1, 1], c='blue')

    if frame >= 3:
        for (i, j) in edges:
            ax.plot(*zip(vertices[i], vertices[j]), c='gray')

    if frame >= 5:
        poly = Polygon(vertices[triangle], closed=True, alpha=0.4, color='orange')
        ax.add_patch(poly)
    return []


ani = FuncAnimation(fig, update, frames=7, init_func=init, blit=False, interval=1000)
plt.show()
