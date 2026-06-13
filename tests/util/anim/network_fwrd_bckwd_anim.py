from animation import *

class NeuralNetworkFwrdBckwd(Scene):
    def construct(self):
        visible_input = 48
        visible_hidden_1 = 32
        visible_hidden_2 = 16
        visible_hidden_3 = 8
        visible_output = 6

        input_layer = self.create_layer(visible_input, LEFT * 5)
        hidden_layer_1 = self.create_layer(visible_hidden_1, LEFT * 3)
        hidden_layer_2 = self.create_layer(visible_hidden_2, ORIGIN)
        hidden_layer_3 = self.create_layer(visible_hidden_3, RIGHT * 3)
        output_layer = self.create_layer(visible_output, RIGHT * 5)

        # === LABELS ===
        input_label = Text("Input Layer (784)", font_size=16).next_to(input_layer[0], UP, buff=1)
        hidden_label_1 = Text("Hidden Layer (256)", font_size=16).next_to(hidden_layer_1[0], UP, buff=1)
        hidden_label_2 = Text("Hidden Layer (128)", font_size=16).next_to(hidden_layer_2[0], UP, buff=1)
        hidden_label_3 = Text("Hidden Layer (64)", font_size=16).next_to(hidden_layer_3[0], UP, buff=1)
        output_label = Text("Output Layer (10)", font_size=16).next_to(output_layer[0], UP, buff=1)

        # === NODES ===

        # === CONNECTIONS (Edges) ===
        input_to_hidden_1 = self.connect_layers(input_layer, hidden_layer_1)
        hidden_1_to_hidden_2 = self.connect_layers(hidden_layer_1, hidden_layer_2)
        hidden_2_to_hidden_3 = self.connect_layers(hidden_layer_2, hidden_layer_3)
        hidden_3_to_output = self.connect_layers(hidden_layer_3, output_layer)

        # === FORWARD PASS ===
        forward_text = Text("MNIST Training Forward Pass", color=GREEN, font_size=28).to_edge(UP)
        self.play(Write(forward_text))

        self.play(*[Write(node) for node in input_layer])
        self.play(*[Write(node) for node in hidden_layer_1])
        self.play(*[Write(node) for node in hidden_layer_2])
        self.play(*[Write(node) for node in hidden_layer_3])
        self.play(*[Write(node) for node in output_layer])

        self.play(*[Create(edge) for edge in input_to_hidden_1])
        self.play(*[Create(edge) for edge in hidden_1_to_hidden_2])
        self.play(*[Create(edge) for edge in hidden_2_to_hidden_3])
        self.play(*[Create(edge) for edge in hidden_3_to_output])

        self.play(Write(input_label))
        self.play(Write(hidden_label_1))
        self.play(Write(hidden_label_2))
        self.play(Write(hidden_label_3))
        self.play(Write(output_label))

        self.animate_signal(input_to_hidden_1, color=GREEN)
        self.animate_signal(hidden_1_to_hidden_2, color=GREEN)
        self.animate_signal(hidden_2_to_hidden_3, color=GREEN)
        self.animate_signal(hidden_3_to_output, color=GREEN)
        self.play(FadeOut(forward_text))
        self.wait(1)

        # === BACKWARD PASS ===
        backward_text = Text("MNIST Training Backward Pass", color=RED, font_size=28).to_edge(UP)
        self.play(Write(backward_text))

        self.animate_signal(hidden_3_to_output, color=RED)
        self.animate_signal(hidden_2_to_hidden_3, color=RED)
        self.animate_signal(hidden_1_to_hidden_2, color=RED)
        self.animate_signal(input_to_hidden_1, color=RED)
        self.play(FadeOut(backward_text))
        self.wait(2)

    def create_layer(self, num_nodes, center):
        spacing = 0.1
        nodes = VGroup(*[
            Sphere(radius=0.03, color=WHITE).move_to(center + DOWN * spacing * (i - num_nodes / 2))
            for i in range(num_nodes)
        ])
        return nodes

    def connect_layers(self, layer1, layer2):
        edges = []
        for node1 in layer1:
            for node2 in layer2:
                line = Line(node1.get_right(), node2.get_left(), stroke_width=1.0, color=GRAY)
                edges.append(line)
        return edges

    def animate_signal(self, edges, color):
        animations = []
        for edge in edges:
            highlight = edge.copy().set_color(color).set_stroke(width=4)
            animations.append(ReplacementTransform(edge, highlight))
            animations.append(ReplacementTransform(highlight, edge))
        self.play(*animations, run_time=2)
