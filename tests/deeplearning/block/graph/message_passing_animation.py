import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import numpy as np
import matplotlib.image as mpimg


def _draw_logo(fig) -> None:
    """
    Draw Logo on the top of the animation frame
    @param fig: Matplotlib figure
    @type fig: Figure
    """
    chart_pos = (0.1, 0.0, 0.4, 0.4)
    img = mpimg.imread('../../../../python/input/Animation_logo.png')
    inset_ax = fig.add_axes(chart_pos)
    inset_ax.imshow(img, alpha=0.8)
    inset_ax.axis('off')


# Define a simple graph
G = nx.Graph()
edges = [(0, 1), (0, 2), (0, 3), (0, 11), (1, 4), (1, 5), (2, 6), (2, 7), (2, 8), (2, 9), (3, 16), (3, 16), (4, 12),
         (4, 13), (6, 14), (6, 15), (5, 16), (1, 10)]
G.add_edges_from(edges)

# Initial node features (scalar values)
features = {i: np.random.rand(3) for i in G.nodes()}
aggregated_features = features.copy()

# Position of nodes
pos = nx.spring_layout(G, dim=2, seed=42, scale=3.0)
# Create figure
fig, ax = plt.subplots(figsize=(10, 7))
colors = []
ax.set_axis_off()


def update(frame):
    ax.clear()
    ax.text(x=-1.75,
            y=-3.6,
            s='Hands-on Geometric Deep Learning',
            ha='left',
            color='blue',
            fontdict={'fontsize': 18, 'fontname': 'Helvetica'})
    ax.text(x=-1.01,
            y=-0.651,
            s="Graph Neural Network\nMessage Passing Tuning\nStep {}".format(frame + 1),
            ha='center',
            color='grey',
            fontdict={'fontsize': 22, 'fontname': 'Helvetica'})
    ax.text(x=-1.0,
            y=-0.65,
            s="Graph Neural Network\nMessage Passing Tuning\nStep {}".format(frame+1),
            ha='center',
            color='black',
            fontdict={'fontsize': 22, 'fontname': 'Helvetica'})
    """
    ax.set_title(y=0.46,
                 x=0.8,
                 label="Message Passing\nTuning Parameters\nStep {}".format(frame+1),
                 fontdict={'fontsize': 22, 'fontname': 'Helvetica'})
    """
    x_shadow_offset = 0.03
    y_shadow_offset = 0.056
    shadow_pos = {k: [v[0]+x_shadow_offset, v[1]+y_shadow_offset] for k, v in pos.items()}
    _ = nx.draw_networkx_nodes(G, shadow_pos, node_size=1200, ax=ax, node_color='grey', edgecolors='grey', linewidths=0)
    _ = nx.draw_networkx_nodes(G, pos, node_size=1200, ax=ax, node_color='blue', edgecolors='black', linewidths=1)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=14)

    # Step 2: Simulate message passing from neighbors
    if frame < len(G.nodes()):
        target = frame
        messages = []

        for neighbor in G.neighbors(target):
            messages.append(features[neighbor])
            # Draw the "message" as an arrow
            x0, y0 = pos[neighbor]
            x1, y1 = pos[target]
            ax.annotate("",
                        xy=(x1, y1),
                        xycoords='data',
                        xytext=(x0, y0),
                        textcoords='data',
                        arrowprops=dict(arrowstyle="->", color='red', lw=4))

        # Step 3: Aggregate (sum) messages and update target feature
        if messages:
            aggregated_features[target] = sum(messages)

    # Step 4: Display node features
    for node in G.nodes():
        f = aggregated_features[node]
        f_str = f'{f[0]:.2f}\n{f[1]:.2f}\n{f[2]:.2f}'
        if target == node:
            color = 'red'
            desc = f'Sum\n'
            font_weight = 'bold'
            font_size = 11
            y_offset = 0.25
        else:
            color = 'black'
            desc = ''
            font_weight = 'regular'
            font_size = 10
            y_offset = 0.2

        ax.text(x=pos[node][0]+0.25,
                y=pos[node][1]-y_offset,
                s=f'{desc}{f_str}',
                fontsize=font_size,
                ha='center',
                fontweight=font_weight,
                color=color)
    ax.set_axis_off()


# Animate
ani = animation.FuncAnimation(fig, update, frames=len(G.nodes()), interval=800, repeat=False)
# plt.show()
fps = 2
ani.save('message_passing.mp4', writer='ffmpeg', fps=fps, dpi=240)
