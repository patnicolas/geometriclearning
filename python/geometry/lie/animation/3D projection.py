import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate N points on a 4D unit hypersphere
def sample_hypersphere_4d(n_points=2000):
    x = np.random.normal(size=(n_points, 4))
    x /= np.linalg.norm(x, axis=1, keepdims=True)  # Normalize to unit sphere
    return x

# Project 4D points to 3D (drop the 4th coordinate)
def project_to_3d(points_4d):
    return points_4d[:, :3]  # Drop the w coordinate


if __name__ == '__main__':
    # Generate and project
    points_4d = sample_hypersphere_4d(n_points=8000)
    points_3d = project_to_3d(points_4d)

    # Plot
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#f0f9ff')
    ax.set_facecolor('#f0f9ff')
    ax.scatter(xs=points_3d[:, 0], ys= points_3d[:, 2], zs=points_3d[:, 1], c=points_3d[:, 2], s=22, alpha=0.8, cmap='rainbow')

    ax.axis('off')  # Hide axes for clarity
    plt.tight_layout()
    plt.show()