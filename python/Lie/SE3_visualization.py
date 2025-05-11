
from Lie.Lie_SE3_group import LieSE3Group, VisualTangentVector
import matplotlib.pyplot as plt
import geomstats.visualization as visualization
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import AnyStr, List, Tuple
import geomstats.backend as gs

class SE3Visualization(LieSE3Group):

    def __init__(self,
                 rot_matrix: np.array,
                 trans_matrix: np.array) -> None:
        super(SE3Visualization, self).__init__(rot_matrix=rot_matrix, trans_matrix=trans_matrix, point_type='vector')

    def visualize(self,
                  visual_tangent_vecs: List[VisualTangentVector],
                  initial_point: np.array = None,
                  scale: (float, float) = (-1.0, 1.0),
                  num_points: int = 8,
                  title: AnyStr = '') -> None:
        """
        Visualize the multiple tangent vectors geodesics from a single SE3 element
        @param visual_tangent_vecs: List of initial tangent vectors
        @type visual_tangent_vecs: List of Numpy array
        @param initial_point: Initial point on a manifold. Identity if not specified
        @type initial_point:  Numpy array
        @param num_points: Number of data points along the geodesics
        @type num_points: int
        @param scale: Location of labels
        @param scale: Tuple[int, int]
        @param title: Title for plot
        @type title: AnyStr
        """
        assert self.point_type == 'vector', \
            f'Cannot visualize SE3 elements for {self.point_type} should be vector'

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        fig.set_facecolor('#F2F9FE')
        ax.set_facecolor('#F2F9FE')

        self.__draw(ax,
                    title,
                    num_points,
                    initial_point,
                    scale,
                    visual_tangent_vecs)
        """
        ax.set_title(y=1.01, label=title, fontdict={'fontsize': 18, 'fontname': 'Helvetica'})
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        ax.set_zlabel('Z', fontsize=14)
        # Initial point is identity if not provided
        initial_point = self.lie_group.identity if initial_point is None else initial_point
        # Normalized input values
        t = gs.linspace(scale[0], scale[1], num_points)
        pts = []
        for idx, initial_tangent_vec in enumerate(visual_tangent_vecs):
            geodesic = self.lie_group.metric.geodesic(
                initial_point=initial_point, initial_tangent_vec=initial_tangent_vec.vec
            )
            pt = geodesic(t)
            pts.append(pt)
            initial_tangent_vec.draw(ax)
        points = np.concatenate(pts, axis=0)
        visualization.plot(points, num_groups=len(visual_tangent_vecs), space="SE3_GROUP")
        """
        plt.show()

    def animate(self,
                visual_tangent_vecs: List[VisualTangentVector],
                initial_point: np.array = None,
                scale: (float, float) = (-1.0, 1.0),
                num_points: int = 8,
                title: AnyStr = '',
                interval: int = 1000,
                fps: int = 20) -> None:
        from matplotlib.animation import FuncAnimation

        assert self.point_type == 'vector', \
            f'Cannot visualize SE3 elements for {self.point_type} should be vector'

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        fig.set_facecolor('#F2F9FE')
        ax.set_facecolor('#F2F9FE')

        # Initial point is identity if not provided
        initial_point = self.lie_group.identity if initial_point is None else initial_point
        # Normalized input values
        text_obj = ax.text(x=0.5, y=-0.5, z=5.0, s=f'0 displacement points', fontdict={'fontsize': 14})

        def update(frame: int) -> None:
            self.__draw(ax,
                        title,
                        num_points,
                        initial_point,
                        scale,
                        visual_tangent_vecs,
                        frame,
                        text_obj)
        ani = FuncAnimation(fig, update, frames=num_points, interval=interval, repeat=False, blit=False)
        plt.show()
        # ani.save('SE3_visualization.mp4', writer='ffmpeg', fps=fps, dpi=240)

    def __draw(self,
               ax: Axes3D,
               title: AnyStr,
               num_points: int,
               initial_point: np.array,
               scale: Tuple[int, int],
               visual_tangent_vecs:  List[VisualTangentVector],
               frame: int == -1,
               text_obj=None):

        ax.set_title(y=1.01, label=title, fontdict={'fontsize': 18, 'fontname': 'Helvetica'})
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        ax.set_zlabel('Z', fontsize=14)

        if text_obj is not None:
            frame_cursor = frame * 5
            if frame_cursor < num_points:
                text_obj.set_text(f'{frame_cursor + 3} displacement points')

        t = gs.linspace(scale[0], scale[1], frame + 2)
        pts = []
        for idx, initial_tangent_vec in enumerate(visual_tangent_vecs):
            geodesic = self.lie_group.metric.geodesic(
                initial_point=initial_point, initial_tangent_vec=initial_tangent_vec.vec
            )
            pt = geodesic(t)
            pts.append(pt)
            initial_tangent_vec.draw(ax)
        points = np.concatenate(pts, axis=0)
        visualization.plot(points, num_groups=len(visual_tangent_vecs), space="SE3_GROUP")
