from manim import *
import networkx as nx

class NetworkXScene(Scene):
    def construct(self):
        # Create a NetworkX graph
        nx_graph = nx.Graph()
        nx_graph.add_edges_from([
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (3, 6),
            (3, 5),
            (5, 6)
        ])
        node_features = {1: 1.0, 2: 3.3, 3: 0.5, 4: -1.0, 5: -2.3, 6: 3.0}

        # Generate layout for nodes
        layout = nx.spring_layout(nx_graph, seed=42)

        # Create a Manim Graph from the NetworkX graph
        graph = Graph(
            vertices=list(nx_graph.nodes),
            edges=list(nx_graph.edges),
            layout="spring",
            vertex_config={"radius": 0.3, "fill_color": BLUE},
            labels=True
        )

        # Animate the graph
        self.play(Create(graph))
        self.wait(0.5)

        feature_labels = {}
        for node in nx_graph.nodes:
            value = node_features[node]
            label = MathTex(f"{value:.1f}", font_size=16)
            label.next_to(graph.vertices[node], DOWN, buff=0.20)
            feature_labels[node] = label
            self.add(label)

        updated_features = {}
        for node in nx_graph.nodes:
            neighbors = list(nx_graph.neighbors(node))
            neighbor_feats = [node_features[n] for n in neighbors]
            if neighbor_feats:
                updated = np.mean(neighbor_feats)
            else:
                updated = node_features[node]
            updated_features[node] = updated

        updated_labels = []
        for node in nx_graph.nodes:
            old_label = feature_labels[node]
            new_val = updated_features[node]
            new_label = MathTex(f"{new_val:.1f}", font_size=16)
            new_label.next_to(graph.vertices[node], DOWN, buff=0.15)
            updated_labels.append(Transform(old_label, new_label))

        self.play(*updated_labels)
        self.wait(2)
        for node in nx_graph.nodes:
            value = updated_features[node]
            _color = interpolate_color(BLUE, RED, value / 4.0)  # Normalize color
            self.play(graph.vertices[node].animate.set_fill(_color), run_time=0.5)


if __name__ == '__main__':
    networkx_scene = NetworkXScene()
    networkx_scene.construct()