from animation import *

class FixedLabels(ThreeDScene):
    def construct(self):
        # Title at the top
        title = Text("SE(3) Animation", font_size=36).to_edge(UP)

        # Corner label
        label = Text("SO(3) Rotation", font_size=24).to_corner(UL)

        # Bottom note
        note = Text("Translation along X", font_size=24).to_edge(DOWN)

        # Mark each as fixed
        self.add_fixed_in_frame_mobjects(title)
        self.wait(2)
        self.add_fixed_in_frame_mobjects(label)
        self.play(Write(label))
        self.wait(3)
        self.play(Unwrite(label))
        self.wait(2)
        self.add_fixed_in_frame_mobjects(note)
        # self.play(Unwrite(note))
        self.wait(2)
        self.play(Unwrite(note))
        self.wait(2)

