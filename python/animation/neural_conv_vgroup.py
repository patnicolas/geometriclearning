


"""
from manim import *

class ConvLayerAnimation(Scene):
    def construct(self):
        # Input feature map (5x5 grid)
        input_grid = VGroup()
        for i in range(5):
            for j in range(5):
                cell = Square(side_length=0.5).set_stroke(GRAY, 1)
                cell.move_to(RIGHT * j * 0.5 + UP * -i * 0.5)
                input_grid.add(cell)
        input_grid.move_to(LEFT * 3)

        # Kernel (3x3)
        kernel = VGroup()
        for i in range(3):
            for j in range(3):
                cell = Square(side_length=0.5).set_stroke(RED, 2)
                cell.move_to(RIGHT * j * 0.5 + UP * -i * 0.5)
                kernel.add(cell)
        kernel.move_to(input_grid[0].get_center())

        # Output feature map (3x3 after valid conv)
        output_grid = VGroup()
        for i in range(3):
            for j in range(3):
                cell = Square(side_length=0.5).set_stroke(BLUE, 1)
                cell.move_to(RIGHT * j * 0.5 + UP * -i * 0.5)
                output_grid.add(cell)
        output_grid.move_to(RIGHT * 3)

        # Labels
        input_label = Text("Input", font_size=24).next_to(input_grid, UP)
        kernel_label = Text("Kernel", font_size=24).next_to(kernel, UP)
        output_label = Text("Output", font_size=24).next_to(output_grid, UP)

        # Activation function label
        activation = MathTex(r"\text{ReLU}(x)", font_size=36).next_to(output_grid, DOWN)

        # Display all
        self.play(FadeIn(input_grid), Write(input_label))
        self.play(FadeIn(kernel), Write(kernel_label))
        self.play(FadeIn(output_grid), Write(output_label))
        self.wait(1)

        # Animate convolution sweep
        for i in range(3):
            for j in range(3):
                # move kernel
                target_index = i * 5 + j
                target_cell = input_grid[target_index]
                self.play(kernel.animate.move_to(target_cell.get_center()), run_time=0.3)
                self.wait(0.1)

        # Apply activation (color flash)
        self.play(Indicate(output_grid, color=YELLOW), Write(activation))
        self.wait(2)

        self.play(*[FadeOut(m) for m in self.mobjects])




"""