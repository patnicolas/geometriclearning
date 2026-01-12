import torch
import torch.nn as nn
from animation import *

class BackPropVisual(Scene):
    def construct(self):
        # Define a simple 2-layer MLP in PyTorch
        model = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

        # Dummy input and forward pass
        x = torch.tensor([[1.0, 2.0]], requires_grad=True)
        y_true = torch.tensor([[1.0]])
        y_pred = model(x)

        # Compute loss and backpropagate
        criterion = nn.MSELoss()
        loss = criterion(y_pred, y_true)
        loss.backward()

        # Visualize the network
        input_nodes = [Dot(LEFT*4 + DOWN + RIGHT*i, color=BLUE) for i in range(2)]
        hidden_nodes = [Dot(RIGHT*i, color=ORANGE) for i in range(2)]
        output_node = Dot(RIGHT*4 + UP, color=GREEN)

        input_labels = [MathTex(f"x_{{{i}}}").next_to(n, DOWN) for i, n in enumerate(input_nodes)]
        hidden_labels = [MathTex(f"h_{{{i}}}").next_to(n, DOWN) for i, n in enumerate(hidden_nodes)]
        output_label = MathTex("\hat{y}").next_to(output_node, UP)

        # Arrows from input to hidden
        input_to_hidden = [Arrow(i.get_center(), h.get_center(), buff=0.1)
                           for i in input_nodes for h in hidden_nodes]

        # Arrows from hidden to output
        hidden_to_output = [Arrow(h.get_center(), output_node.get_center(), buff=0.1)
                            for h in hidden_nodes]

        # Animate network structure
        self.play(*[FadeIn(n) for n in input_nodes + hidden_nodes + [output_node]])
        self.play(*[Write(lbl) for lbl in input_labels + hidden_labels + [output_label]])
        self.play(*[Create(a) for a in input_to_hidden + hidden_to_output])
        self.wait(1)

        # Show forward pass
        forward_dots = [n.copy().set_color(YELLOW).scale(1.2) for n in input_nodes + hidden_nodes + [output_node]]
        self.play(*[Indicate(d) for d in forward_dots])
        self.wait(1)

        # Show gradients on weights (simulate as red arrows)
        grad_arrows = [a.copy().set_color(RED).set_opacity(0.6) for a in hidden_to_output + input_to_hidden]
        self.play(*[GrowArrow(g) for g in grad_arrows])
        self.wait(2)

        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects])
