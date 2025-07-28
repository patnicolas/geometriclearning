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

from typing import AnyStr, Tuple, Dict, List
__all__ = ['DrawTree']

class DrawTree(object):
    offset_levels = (0.07, 0.27, 0.47)
    icons = {'Geometry': '',
             'Linear Algebra': '',
             'Topology': '',
             'Manifolds': '',
             'Algebra & Groups': '',
             'lie Groups': '',
             'Equivariant Networks': '',
             'Machine Learning': '',
             'Graph Neural Networks': '',
             'Manifold-based Models': ''
             }
    colors = ['black', 'red', 'blue', 'green', 'grey']
    font_sizes = [17, 15, 13, 11, 10]

    def __init__(self, tree_data: Dict[AnyStr, AnyStr], fig_size: Tuple[int, int], dx: float, dy: float) -> None:
        self.tree_data = tree_data
        self.dx = dx
        self.dy = dy
        fig, self.ax = plt.subplots(figsize=fig_size)

    @staticmethod
    def __get_color_font_size(level: int) -> Tuple[AnyStr, int]:
        return DrawTree.colors[level], DrawTree.font_sizes[level]

    def draw(self) -> None:
        def draw_tree(tree: Dict[AnyStr, AnyStr], x: float, y: List[float], level: int) -> None:
            for i, (key, subtree) in enumerate(tree.items()):
                current_x = x + level * self.dx
                current_y = y[0]  # Use mutable y so it updates across recursive calls
                color, font_size = DrawTree.__get_color_font_size(level)

                # Draw the node
                if level == 0:
                    self.ax.text(current_x + 0.05, current_y, key, va='center', fontsize=17, fontweight='bold')
                else:
                    self.ax.plot([current_x - self.dx + 0.02, current_x - 0.01], [current_y, current_y], 'k-', color='grey')
                    if level > 1:
                        self.ax.plot([DrawTree.offset_levels[0], DrawTree.offset_levels[0]],
                                [current_y + self.dy, current_y],
                                'k-',
                                color='grey')
                    if level > 2:
                        self.ax.plot([DrawTree.offset_levels[1], DrawTree.offset_levels[1]],
                                [current_y + self.dy, current_y],
                                'k-')
                    self.ax.plot([current_x - self.dx + 0.02, current_x - self.dx + 0.02],
                            [current_y + self.dy, current_y],
                            'k-',
                            color='grey')
                    icon = "\U000025C6"
                    self.ax.text(current_x + 0.05,
                                 current_y,
                                 f"{icon} {key}",
                                 va='center',
                                 fontsize=font_size,
                                 fontweight='normal',
                                 color=color,
                                 fontfamily='DejaVu Sans')

                y[0] -= self.dy  # Update Y-position for the next sibling
                # Recurse for children
                draw_tree(subtree, x, y, level + 1)
        # Draw the tree
        draw_tree(tree_input, x=0.05, y=[1.0], level=0)

        self.ax.axis('off')
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    import matplotlib.pyplot as plt


    tree_input = {
        'Geometric Deep Learning': {
            'Geometry': {
                'Differential Geometry': {},
                'Manifolds': {},
                'Riemannian Metrics': {},
                'Geodesics': {}
            },
            'Topology': {
                'Topology Data Analysis': {
                    'Persistent Homology': {},
                    'Simplicial Complexes': {},
                    'Combinatorial Complexes': {},
                    'Filtration': {}
                },
                'Homology': {
                    'Cohomology': {},
                    'Sheaf Theory': {}
                }
            }
        }
    }

    draw_tree = DrawTree(tree_data=tree_input, fig_size=(6, 10), dx=0.2, dy=0.2)
    draw_tree.draw()

    """
    offset_level_1 = 0.07
    offset_level_2 = 0.27

    # Recursive function to draw the tree
    def draw_tree(ax, tree, x, y, x_offset, y_step, level=0):
        for i, (key, subtree) in enumerate(tree.items()):
            current_x = x + level * x_offset
            current_y = y[0]  # Use mutable y so it updates across recursive calls

            # Draw the node
            if level == 0:
                ax.text(current_x+0.05, current_y, key, va='center', fontsize=17, fontweight='bold')
            else:
                color = 'red'
                font_size = 15
                ax.plot([current_x - x_offset + 0.02, current_x - 0.01], [current_y, current_y], 'k-', color='grey')
                if level > 1:
                    color = 'blue'
                    font_size = 13
                    ax.plot([offset_level_1, offset_level_1], [current_y + y_step, current_y], 'k-', color='grey')
                if level > 2:
                    color = 'green'
                    font_size = 11
                    ax.plot([offset_level_2, offset_level_2], [current_y + y_step, current_y], 'k-')
                ax.plot([current_x - x_offset + 0.02, current_x - x_offset + 0.02], [current_y + y_step, current_y],
                        'k-', color='grey')
                icon = "\U000025C6"
                ax.text(current_x + 0.05, current_y, f"{icon} {key}", va='center', fontsize=font_size, fontweight='normal', color=color, fontfamily='DejaVu Sans')

            y[0] -= y_step  # Update Y-position for the next sibling

            # Recurse for children
            draw_tree(ax, subtree, x, y, x_offset, y_step, level + 1)


    # Create figure and axes
    fig, ax = plt.subplots(figsize=(4, 5))
    initial_y = [1.0]  # Mutable to track vertical position across recursion

    # Draw the tree
    draw_tree(ax, tree_input, x=0.05, y=initial_y, x_offset=0.2, y_step=0.2)

    ax.axis('off')
    plt.tight_layout()
    plt.show()



    ---------------------------------
    import matplotlib.pyplot as plt
    import networkx as nx

    tree_input = {
        'name': 'Geometric Deep Learning',
        'children': [
            {'name': 'Geometry', 'children': [
                {'name': 'Differential Geometry'},
                {'name': 'Manifolds'},
                {'name': 'Riemannian Metrics'},
                {'name': 'Geodesics'},
            ]},
            {'name': 'Topology', 'children': [
                {'name': 'Topology Data Analysis', 'children': [
                    {'name': 'Persistent Homology'},
                    {'name': 'Simplicial Complexes'},
                    {'name': 'Combinatorial Complexes'},
                    {'name': 'Filtration'},
                ]},
                {'name': 'Homology'},
                {'name': 'Cohomology'},
                {'name': 'Sheaf Theory'},
            ]},
        ]
    }


    # Recursive function to add nodes/edges
    def add_nodes_edges(tree, G=None, parent=None):
        if G is None:
            G = nx.DiGraph()
        name = tree["name"]
        G.add_node(name)
        if parent:
            G.add_edge(parent, name)
        for child in tree.get("children", []):
            add_nodes_edges(child, G, name)
        return G

    # Vertical (top-down) hierarchy layout
    def vertical_hierarchy_pos(G, root=None, x=0, y=0, x_spacing=1.5, y_spacing=1.5, pos=None, depth=0):
        if pos is None:
            pos = {}
        pos[root] = (x, y)
        children = list(G.successors(root))
        if children:
            dx = x_spacing * (len(children) - 1)
            next_x = x - dx / 2
            for child in children:
                pos = vertical_hierarchy_pos(G,
                                             root=child,
                                             x=next_x,
                                             y=y - y_spacing,
                                             x_spacing=x_spacing,
                                             y_spacing=y_spacing,
                                             pos=pos,
                                             depth=depth + 1)
                next_x += x_spacing
        return pos


    # Build and layout
    tree_graph = add_nodes_edges(tree_input)
    pos = vertical_hierarchy_pos(tree_graph, root="Geometric Deep Learning")

    # Draw graph
    plt.figure(figsize=(8, 6))
    nx.draw(tree_graph, pos, with_labels=True, arrows=False,
            node_size=3000, node_color='lightcyan', font_size=10)
    plt.gca().invert_yaxis()  # So root is on top
    plt.axis('off')
    plt.show()
    """


