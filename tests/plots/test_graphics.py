import unittest
import random


class GraphicsTest(unittest.TestCase):

    def test_drop_out_visualization(self):
        import numpy as np
        import matplotlib.pyplot as plt

        # Generate random heatmap data
        x, y = np.meshgrid(np.arange(10), np.arange(10))  # 5x5 grid
        x = x.flatten()
        y = y.flatten()
        z = np.zeros_like(x)  # Base height
        dx = dy = np.ones_like(z)  # Bar width
        dz = [random.random() if random.random() > 0.75 else 0 for _ in range(len(x))]

        # dz = np.random.rand(len(x))  # Heatmap values

        # Create 3D figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot 3D bars
        ax.bar3d(x, y, z, dx, dy, dz, shade=True, color="darkgrey")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.grid(False)
        plt.title("3D Heatmap (Bar Plot)")
        plt.show()
        self.assertTrue(True)

    def test_batch_norm_visualization(self):
        import numpy as np
        import matplotlib.pyplot as plt

        X = np.linspace(-2, 2, 30)
        Y = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(X, Y)
        Z = np.exp(0.025*X*Y)  # Simulating a feature map before normalization

        fig = plt.figure(figsize=(8, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap="BuPu", edgecolor="none")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.grid(False)
        plt.title("BatchNorm Visualization")
        plt.show()
        self.assertTrue(True)

    def test_activation_visualization(self):
        import numpy as np
        import matplotlib.pyplot as plt

        X = np.linspace(-2, 2, 30)
        Y = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(X, Y)
        Z = np.tanh(X+Y)  # Simulating a feature map before normalization

        fig = plt.figure(figsize=(8, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Z, Y, X, cmap="BuPu", edgecolor="none")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.grid(False)
        plt.title("BatchNorm Visualization")
        plt.show()
        self.assertTrue(True)

    def test_activation_visualization_wireframe(self):
        import numpy as np
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        X = np.linspace(-2, 2, 30)
        Y = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(X, Y)
        Z = np.tanh(X+Y)  # Simulating a feature map before normalization

        ax.plot_wireframe(X, Y, Z, color='blue')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.grid(False)

        plt.title("3D Heatmap (Wireframe)")
        plt.show()
        self.assertTrue(True)


