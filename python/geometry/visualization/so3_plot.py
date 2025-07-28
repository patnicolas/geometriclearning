__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

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

from geometry.visualization.manifold_plot import ManifoldPlot
from typing import List, Optional
import numpy as np



class SO3Plot(ManifoldPlot):
    def __init__(self, manifold_points: np.array) -> None:
        """
        Constructor for plotting Special Orthogonal Group in dimension 3
        @type manifold_points: List of Numpy arrays representing points on the manifold
        @type manifold_points: List
        """
        super(SO3Plot, self).__init__(manifold_points)

    def show(self, extra_components: Optional[np.array] = None) -> None:
        """
        Plot the SO3 rotation 3x3 matrices on a 3D plot with rho =1, theta and phi
        @param extra_components: Optional components to be added to the plot
        @type extra_components: Numpy array
        """
        import matplotlib.pyplot as plt
        import geomstats.backend as gs

        num_points = len(self.manifold_points)

        # Build the 3D grid
        theta = np.linspace(start=0.0, stop=np.pi, num=num_points)
        phi = np.linspace(start=0.0, stop=2 * np.pi, num=num_points)
        theta, phi = np.meshgrid(theta, phi)

        # Create the coordinate for the matrix
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for i, rotation in enumerate(self.manifold_points):
            # Apply the rotation to the sphere points
            points = np.stack(arrays=[x.ravel(), y.ravel(), z.ravel()], axis=1)
            r_points = gs.matmul(rotation, points.T).T  # Apply rotation

            # Reshape rotated points for plotting
            x_r = r_points[:, 0].reshape(num_points, num_points)
            y_r = r_points[:, 1].reshape(num_points, num_points)
            z_r = r_points[:, 2].reshape(num_points, num_points)

            # Plot the rotated sphere
            ax.plot_surface(
                x_r, y_r, z, alpha=0.6, edgecolor='k', linewidth=0.3, label=f"Rotation {i + 1}"
            )

        ManifoldPlot._create_legend(title='SO(3) Rotations acting on the Unit Sphere', ax=ax)
        plt.show()

