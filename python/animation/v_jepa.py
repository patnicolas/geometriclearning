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

import sys
import os
from manim import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class JEPAPrediction(Scene):
    def construct(self):
        # 1. Create Input Image (Simulated as a grid)
        grid = VGroup(*[Square(side_length=0.5) for _ in range(16)]).arrange_in_grid(rows=4, cols=4)
        input_label = Text("Input Image", font_size=24).next_to(grid, UP)

        # 2. Masking Step
        mask = Square(side_length=1.1, color=RED, fill_opacity=0.5).move_to(grid[5])
        mask_label = Text("Masked Region (Target)", color=RED, font_size=18).next_to(mask, RIGHT)

        self.play(Create(grid), Write(input_label))
        self.wait(1)
        self.play(FadeIn(mask), Write(mask_label))
        self.wait(1)

        # 3. Define the Encoders (Latent Space)
        latent_box = RoundedRectangle(corner_radius=0.2, height=3, width=5, color=BLUE)
        latent_label = Text("Embedding Space (Latent)", font_size=24).next_to(latent_box, UP)

        context_enc = Text("Context Encoder", font_size=20, color=YELLOW).shift(LEFT * 1.5)
        target_enc = Text("Target Encoder", font_size=20, color=GREEN).shift(RIGHT * 1.5)

        # 4. Animate the Encoding
        self.play(
            grid.animate.scale(0.5).to_edge(LEFT),
            FadeOut(input_label, mask_label, mask),
            Create(latent_box),
            Write(latent_label)
        )

        # Represent features as dots in the embedding space
        context_dot = Dot(color=YELLOW).move_to(context_enc.get_bottom() + DOWN * 0.5)
        target_dot = Dot(color=GREEN).move_to(target_enc.get_bottom() + DOWN * 0.5)

        self.play(Write(context_enc), Write(target_enc))
        self.play(FadeIn(context_dot), FadeIn(target_dot))

        # 5. The Prediction Step
        predictor_label = Text("Predictor", font_size=18, color=WHITE).move_to(ORIGIN + UP * 0.5)
        prediction_arrow = Arrow(context_dot.get_right(), target_dot.get_left(), buff=0.2)

        predicted_dot = Dot(color=ORANGE).move_to(target_dot.get_center())

        self.play(Write(predictor_label))
        self.play(GrowArrow(prediction_arrow))
        self.play(TransformFromCopy(context_dot, predicted_dot))

        # 6. Show the Loss (Comparison)
        loss_brace = BraceBetweenPoints(predicted_dot.get_right(), target_dot.get_right(), RIGHT)
        # loss_text = loss_brace.get_text("L2 / Cosine Loss", {'font_size': 18})
        loss_text = loss_brace.get_text("L2 / Cosine Loss")
        self.play(Create(loss_brace), Write(loss_text))
        self.wait(2)


if __name__ == '__main__':
    jepa_prediction = JEPAPrediction()
    jepa_prediction.construct()