import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Create a simple undirected graph
G = nx.erdos_renyi_graph(n=14, p=0.4, seed=42, directed=False)
# G = nx.Graph()
# G = nx.path_graph(9)
# edges = [(0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (2, 3), (3, 6), (0, 7), (7, 8)]
# G.add_edges_from(edges)

# Assign random 3D positions to nodes
pos = nx.spring_layout(G, dim=3, seed=5,  scale=20.0)
# Extract node coordinates
# xyz = np.array([pos[i] for i in G.nodes])

# Set up 3D plot
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_title("3D GNN Message Passing Animation")
ax.set_axis_off()

# Initial node features (scalar values)
features = {i: np.random.rand(3) for i in G.nodes()}
aggregated_features = features.copy()

# Draw static nodes
# nodes_scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='skyblue', s=100, edgecolor='k', depthshade=True)

# Draw static edges

edge_lines = []
for i, j in G.edges:
    line = ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], [pos[i][2], pos[j][2]], 'gray', lw=2.0, alpha=0.5)
    edge_lines.append(line)


# Message passing animation (highlight one edge at a time)
edges = list(G.edges)
arrows = []

def update(frame):
    # Clear previous arrows
    for arrow in arrows:
        arrow.remove()
    arrows.clear()
   # _ = nx.draw_networkx_nodes(G, pos, node_size=600, ax=ax, node_color='blue', edgecolors='black', linewidths=1)

    i, j = edges[frame % len(edges)]
    xi, yi, zi = pos[i]
    xj, yj, zj = pos[j]
    print(f'Frame: {frame} xi={xi} yi={yi} zi={zi}')

    # Draw a message arrow from i to j
    arrow = ax.quiver(
        xi, yi, zi,
        xj - xi, yj - yi, zj - zi,
        color='red', arrow_length_ratio=0.2, linewidth=2
    )
    arrows.append(arrow)
    ax.set_title(f"Message from Node {i} to Node {j}")

ani = FuncAnimation(fig, update, frames=len(edges)*2, interval=1000, repeat=True)

plt.show()
