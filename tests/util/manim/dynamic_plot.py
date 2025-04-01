import matplotlib.pyplot as plt
from animation import *
from typing import List, AnyStr


def generate_data(num_samples: int) -> np.array:
    import math
    import random
    values = np.array([x * math.sin(x) + random.random() for x in range(num_samples)])
    values = values.reshape((-1, 32))
    print(f'x shape {values.shape}')
    return values


class DynamicPlot(Scene):
    data = generate_data(4*32)

    def construct(self) -> None:
        frames = self.__record()
        for frame in frames:
            self.play(FadeIn(frame), run_time=0.3)
            self.wait(0.2)
            self.remove(frame)

    def __record(self) -> List[ImageMobject]:
        frames = []
        num_frames = DynamicPlot.data.shape[0]
        print(f'num frames {num_frames}')
        for epoch in range(num_frames):
            plt.plot(range( DynamicPlot.data.shape[1]), DynamicPlot.data[epoch])
            fname = f'frame_{epoch}.png'
            plt.savefig(fname)
            plt.close()
            frames.append(ImageMobject(fname).scale(2))
        return frames

