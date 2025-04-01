from animation import *
import numpy as np


class ManimMLPDemo1(Scene):
    def construct(self):
        # Network architecture
        layers = [3, 4, 3, 1]
        layer_spacing = 2.5
        neuron_radius = 0.25

        # Positioning
        neuron_groups = []
        layer_positions = []

        for i in range(len(layers)):
            x = i * layer_spacing - 3  # horizontal shift
            layer_positions.append(x)

        # Draw neurons
        for i, (num_neurons, x) in enumerate(zip(layers, layer_positions)):
            neurons = VGroup()
            for j in range(num_neurons):
                y = j * 1 - (num_neurons - 1) * 0.5
                neuron = Circle(radius=neuron_radius, color=BLUE).move_to([x, y, 0])
                neurons.add(neuron)
                self.add(neuron)
            neuron_groups.append(neurons)

        self.wait(1)

        # Draw connections with weights
        connections = VGroup()
        weight_labels = VGroup()

        np.random.seed(0)  # For reproducibility

        for l in range(len(neuron_groups) - 1):
            for i, n1 in enumerate(neuron_groups[l]):
                for j, n2 in enumerate(neuron_groups[l + 1]):
                    weight = np.round(np.random.uniform(-1, 1), 2)
                    color = interpolate_color(RED, GREEN, (weight + 1) / 2)
                    line = Line(n1.get_right(), n2.get_left(), stroke_width=2 + abs(weight) * 2, color=color)
                    connections.add(line)

                    # Weight label
                    mid = line.get_center()
                    label = Text(str(weight), font_size=20).move_to(mid + 0.1 * UP)
                    weight_labels.add(label)

        self.play(Create(connections), Write(weight_labels))
        self.wait(1)

        # Simulate forward pass
        for l in range(len(neuron_groups) - 1):
            for n1 in neuron_groups[l]:
                for n2 in neuron_groups[l + 1]:
                    signal = Dot(color=YELLOW).move_to(n1.get_right())
                    self.add(signal)
                    self.play(signal.animate.move_to(n2.get_left()), run_time=0.3)
                    self.remove(signal)
            self.wait(0.2)

        # Highlight output
        final = neuron_groups[-1][0]
        self.play(final.animate.set_color(GREEN).scale(1.2))
        self.wait(2)


if __name__ == '__main__':
    ManimMLPDemo1().construct()