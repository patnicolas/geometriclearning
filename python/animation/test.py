
from manim import *

class Test(ThreeDScene):
    def construct(self):
        tracker = ValueTracker(0.0)
        m = 2
        size = 5

        activation = "Softmax" if m == 2 else "ReLU"
        text = MathTex(rf'\textbf{{{size}}} \ units \\ \textbf{{{activation}', font_size=48)
        self.play(Write(text))

        circle = Circle(radius=1.0, color=GREEN).shift(LEFT)
        rectangle = Rectangle(color=RED, height=1.0, width=2.0).shift(RIGHT*2)

        def circle_color_updater(mobs: Mobject) -> None:
            tracker_value = int(tracker.get_value())
            if tracker_value >= 4:
                mobs.set_color(RED)

        def rectangle_color_updater(mobs: Mobject) -> None:
            tracker_value = int(tracker.get_value())
            if tracker_value < 3:
                mobs.to_corner(UL, buff=0.5)
            elif 3 <= tracker_value < 9:
                mobs.set_color(BLUE)
            elif 9 <= tracker_value < 14:  #12
                mobs.to_corner(UR, buff=0.5)
                mobs.set_color(WHITE)
            else:     # 15
                mobs.to_corner(DR, buff=0.5)
                mobs.set_color(GREEN)

        self.play(Write(circle), run_time=0.1)
        self.play(Write(rectangle), run_time=0.1)
        circle.add_updater(circle_color_updater)
        rectangle.add_updater(rectangle_color_updater)
        self.play(tracker.animate.set_value(18), run_time=18)
        self.wait()


if __name__ == '__main__':
    scene = Test()
    scene.construct()