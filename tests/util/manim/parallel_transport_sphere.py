from animation import *
import numpy as np

class ParallelTransportOnSphere(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)

        # Create the sphere
        sphere = Sphere(radius=2, resolution=(30, 30), fill_opacity=0.2, checkerboard_colors=[BLUE_D, BLUE_E])
        self.add(sphere)

        # Starting point on sphere (lat, lon)
        theta0 = PI / 2  # Equator
        phi0 = 0

        # Path along the equator (geodesic)
        n_steps = 50
        arc = []
        for i in range(n_steps + 1):
            phi = phi0 + i * PI / n_steps  # half-circle
            x = 2 * np.sin(theta0) * np.cos(phi)
            y = 2 * np.sin(theta0) * np.sin(phi)
            z = 2 * np.cos(theta0)
            arc.append(np.array([x, y, z]))

        # Draw the geodesic
        geodesic = VMobject(color=YELLOW)
        geodesic.set_points_smoothly(arc)
        self.play(Create(geodesic), run_time=2)

        # Initial tangent vector at point 0 (tangent to sphere)
        point = Dot3D(point=arc[0], color=WHITE)
        self.add(point)

        # Initial tangent vector: perpendicular to radius and geodesic direction
        def tangent_vector(p, direction):
            normal = normalize(p)
            tangent = normalize(np.cross(direction, normal))
            return tangent

        # Create the arrow (vector)
        direction = arc[1] - arc[0]
        tangent = tangent_vector(arc[0], direction)
        arrow = Arrow3D(
            start=arc[0],
            end=arc[0] + 0.6 * tangent,
            color=RED,
            thickness=0.02
        )
        self.add(arrow)

        # Animate parallel transport
        for i in range(1, len(arc)):
            p_prev = arc[i - 1]
            p_curr = arc[i]
            direction = p_curr - p_prev
            tangent = tangent_vector(p_curr, direction)

            new_arrow = Arrow3D(
                start=p_curr,
                end=p_curr + 0.6 * tangent,
                color=RED,
                thickness=0.02
            )

            # Move the arrow to the next point
            self.play(Transform(arrow, new_arrow), run_time=0.2)

        self.wait(2)


if __name__ == '__main__':
    instance = ParallelTransportOnSphere()
    instance.construct()
