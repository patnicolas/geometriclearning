import unittest
from typing import AnyStr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm


class PlottingPlay(object):
    def __init__(self, plotting_engine: AnyStr) -> None:
        self.plotting_engine = plotting_engine

    @unittest.skip('Ignored')
    def normal_distribution_3d_plot_play(self)-> None:
        title = f'3D Visualization of a 2D Normal Distribution - {self.plotting_engine}'
        if self.plotting_engine == 'seaborn':
            sns.set_theme(style="whitegrid")

        # 1. Create a grid of x and y coordinates
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)

        # 2. Calculate the 2D Normal Distribution PDF
        # Standard deviation (sigma) = 1, Mean (mu) = 0
        Z = (1 / (2 * np.pi)) * np.exp(-0.5 * (X ** 2 + Y ** 2))

        # 3. Initialize the 3D plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # 4. Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.9)

        # 5. Add labels and customization
        ax.set_title(title)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Probability Density')
        fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar for scale
        plt.show()

    def violin_plot_play(self):
        import pandas as pd

        normal_df = pd.DataFrame({
            'Cluster1': np.random.normal(0, 1.0, 100),
            'Cluster2': np.random.normal(0, 2.0, 100),
            'Cluster3': np.random.normal(0, 3.0, 100),
            'Cluster4': np.random.normal(0, 4.0, 100)
        })

        fig, ax = plt.subplots()
        if self.plotting_engine == 'seaborn':
            sns.violinplot(normal_df, palette='muted')
        else:
            data = normal_df.to_numpy()
            parts = ax.violinplot(data, showmeans=True, showmedians=True)
            # Manually style the violins
            for pc in parts['bodies']:
                pc.set_facecolor('royalblue')
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
            plt.xlabel('Cluster')
            plt.ylabel('Value')
        ax.set_title(f"Violin Plot in {self.plotting_engine}")
        plt.show()

    @unittest.skip('Ignored')
    def scatter_3d_plot_play(self) -> None:
        def get_palette():
            if self.plotting_engine == 'seaborn':
                return sns.color_palette("bright")  # Use a vibrant Seaborn palette
            else:
                return ['royalblue', 'crimson', 'forestgreen']

        def create_cluster(center, num_points=100, spread=0.5):
            return center + np.random.normal(scale=spread, size=(num_points, 3))

        title = f'3D Scatter Plot of Data Clusters - {self.plotting_engine}'
        if self.plotting_engine == 'seaborn':
            sns.set_theme(style="whitegrid")

        # 1. Setup the figure and 3D axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 2. Generate synthetic cluster data
        cluster1 = create_cluster(center=[1, 1, 1])
        cluster2 = create_cluster(center=[4, 4, 4])
        cluster3 = create_cluster(center=[1, 4, 1])

        # 3. Plot each cluster with unique colors and labels
        colors = get_palette()
        ax.scatter(cluster1[:, 0], cluster1[:, 1], cluster1[:, 2], color=colors[0], label='Cluster A', s=30, alpha=0.6)
        ax.scatter(cluster2[:, 0], cluster2[:, 1], cluster2[:, 2], color=colors[1], label='Cluster B', s=30, alpha=0.6)
        ax.scatter(cluster3[:, 0], cluster3[:, 1], cluster3[:, 2], color=colors[2], label='Cluster C', s=30, alpha=0.6)

        # 4. Customizing the view and labels
        ax.set_xlabel('Feature X')
        ax.set_ylabel('Feature Y')
        ax.set_zlabel('Feature Z')
        ax.set_title(title)
        ax.legend()

        # Optional: Set the initial camera angle
        ax.view_init(elev=20, azim=45)
        plt.show()


if __name__ == '__main__':
    plotting_play = PlottingPlay('seaborn')
    plotting_play.violin_plot_play()