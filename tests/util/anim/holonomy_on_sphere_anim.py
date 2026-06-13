from animation import *
import numpy as np

class HolonomyOnSphere(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=40 * DEGREES)
        radius = 2

        # Create sphere
        sphere = Sphere(radius=radius, resolution=(30, 30), fill_opacity=0.2, checkerboard_colors=[GREY_D, GREY_E])
        self.add(sphere)

        # Define three points on the sphere forming a triangle
        A = radius * np.array([1, 0, 0])  # Equator, 0°
        B = radius * np.array([0, 1, 0])  # Equator, 90°
        C = radius * np.array([0, 0, 1])  # North pole

        # Draw triangle path (geodesic arcs)
        arc_AB = ArcBetweenPoints(A, B, color=YELLOW, radius=radius)
        arc_BC = ArcBetweenPoints(B, C, color=YELLOW, radius=radius)
        arc_CA = ArcBetweenPoints(C, A, color=YELLOW, radius=radius)

        triangle_path = VGroup(arc_AB, arc_BC, arc_CA)
        self.play(Create(triangle_path), run_time=3)

        # Initial vector at A (perpendicular to radius and AB direction)
        def get_tangent(p, next_p):
            normal = normalize(p)
            direction = next_p - p
            tangent = normalize(np.cross(direction, normal))
            return tangent

        vector = Arrow3D(
            start=A,
            end=A + 0.5 * get_tangent(A, B),
            color=RED,
            thickness=0.025
        )
        self.add(vector)
        self.wait(1)

        # Helper: transport vector along an arc
        def transport_vector_along_arc(arc, steps=30):
            points = [arc.point_from_proportion(i / steps) for i in range(steps + 1)]
            for i in range(1, len(points)):
                p_prev = points[i - 1]
                p_curr = points[i]
                tangent = get_tangent(p_curr, points[min(i + 1, steps)])
                new_vector = Arrow3D(
                    start=p_curr,
                    end=p_curr + 0.5 * tangent,
                    color=RED,
                    thickness=0.035
                )
                self.play(Transform(vector, new_vector), run_time=0.12)

        # Transport around triangle
        transport_vector_along_arc(arc_AB)
        transport_vector_along_arc(arc_BC)
        transport_vector_along_arc(arc_CA)

        self.wait(2)

        # Final vector is rotated compared to initial direction
        self.play(vector.animate.set_color(GREEN))
        self.wait(2)