__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

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

from manim import *
from manim.utils.color.SVGNAMES import LIGHTGREY

"""
Instead of simple arrows, you can animate flowing small dots ("messages").
Use ValueTracker to animate the node features dynamically changing.
Show parallel message passing: multiple arrows at once, then aggregate.
For "aggregation", you could even animate sum/average visually (e.g., using Plus or Mean symbols).

"""

class GNNMessagePassingWithPackets(Scene):
    def construct(self):
        packet_size = 0.12
        # Create a simple graph
        graph = Graph(
            vertices=["A", "B", "C", "D", "E", "F"],
            edges=[("A", "C"), ("B", "C"), ("D", "C"), ("C", "E"), ("F", "E")],
            layout="spectral",
            labels=True,
            vertex_config={"fill_color": LIGHTGREY},
            edge_config={"fill_color": LIGHTGREY},
        )
        self.play(Create(graph))

        # Shortcuts to node mobjects
        labeled_dot: LabeledDot = graph.vertices["A"]
        node_a = graph.vertices["A"]
        node_b = graph.vertices["B"]
        node_c = graph.vertices["C"]
        node_d = graph.vertices["D"]
        node_e = graph.vertices["E"]
        node_f = graph.vertices["F"]

        # Create message packets (small dots)
        packet_ac = Dot(radius=packet_size, color=BLACK).move_to(node_a.get_center())
        packet_bc = Dot(radius=packet_size, color=BLACK).move_to(node_b.get_center())
        packet_dc = Dot(radius=packet_size, color=BLACK).move_to(node_d.get_center())
        packet_ce = Dot(radius=packet_size, color=RED).move_to(node_c.get_center())
        packet_fe = Dot(radius=packet_size, color=RED).move_to(node_f.get_center())

        # Show packets being created
        self.add(packet_ac, packet_bc, packet_dc, packet_ce, packet_fe)

        # Move packets from A->B and C->B
        move_ab = packet_ac.animate.move_to(node_c.get_center())
        move_bc = packet_bc.animate.move_to(node_c.get_center())
        move_dc = packet_dc.animate.move_to(node_c.get_center())
        move_ce = packet_ce.animate.move_to(node_e.get_center())
        move_fe = packet_fe.animate.move_to(node_e.get_center())

        # Play movements simultaneously
        self.play(move_ab, move_bc, move_dc, move_ce, move_fe, run_time=4)

        # Pulse effect to show aggregation
        self.play(node_c.animate.scale(1.2).set_color(RED))
        self.wait(0.2)
        self.play(node_e.animate.scale(1/1.2).set_color(LIGHTGREY))

        # Remove packets after delivery
        # self.remove(packet_ab, packet_cb)

        # Update node label to reflect updated feature
        new_label = MathTex("h'_B").move_to(node_b)
        self.play(Transform(graph.vertices["B"], new_label))

        self.wait(2)

