import unittest
import jax
import jax.numpy as jnp
from typing import Any
import logging

from control.extended_kalman import ExtendedKalmanFilter

class ExtendedKalmanTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_jax_jacobiana(self):
        def f(_xy: jnp.array) -> jnp.array:
            x, y = _xy
            return jnp.array([x**2+y, jnp.sin(x)*y])

        def g(_xy: jnp.array) -> jnp.array:
            x, y = _xy
            return jnp.array([x ** 2, x * y])

        try:
            J_func_F: Any = jax.jacfwd(f)
            x0 = 1.0
            y0 = 1.0
            xy = jnp.array([x0, y0])
            J = J_func_F(xy)
            logging.info(f'Jacobian (x^2+y, y.sin(x)\n{J}')

            J_func_G: Any = jax.jacfwd(g)
            x0 = 2.0
            y0 = 3.0
            xy = jnp.array([x0, y0])
            J = J_func_G(xy)
            logging.info(f'Jacobian (x^2, x.y)\n{J}')
            self.assertTrue(True)
        except RuntimeWarning as trw:
            logging.warning(trw)
            self.assertTrue(False)
        except RuntimeError as e:
            logging.error(e)
            self.assertTrue(False)

    def test_extended(self):
        import random

        def motion_2d(xyv: jnp.array) -> jnp.array:
            x, y, vx, vy = xyv
            dt = ExtendedKalmanFilter.dt
            return jnp.array([x + vx*dt, y + vy*dt, vx*dt, vy*dt])

        def observed(xy: jnp.array) -> jnp.array:
            return xy

        def generator(t: float) -> jnp.array:
            xt = t + 0.5*(random.uniform(0, 1)-0.5)
            yt = t + 0.5*(random.uniform(0, 1)-0.5)
            return jnp.array([xt, yt, 0.1, 0.2])

        try:
            x0 = jnp.array([0.0, 0.0, 0.2, 0.2])
            P0 = jnp.eye(4, 4)
            cov_means = (0.6, 0.6)
            ekf = ExtendedKalmanFilter.build(x0, motion_2d, observed, P0, cov_means)
            logging.info(str(ekf))
            num_points = 50
            estimation = ekf.simulate(num_points,
                                      lambda t: generator(t),
                                      jnp.array([0.1, 0.3, 0.1, 0.2]))
            for r in estimation:
                logging.info(f'z observed:\n{r[0]}\nState:\n{r[1]}')

            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(121)
            x: jnp.array = jnp.linspace(0, num_points*ExtendedKalmanFilter.dt, num_points)
            _x = x.tolist()
            for e in estimation:
                logging.info(f'e[0]:\n{e[0]}\ne[1]:\n{e[1]}')

            ax1.plot(_x, [e[0][0] for e in estimation])
            ax1.plot(_x, [e[1][0] for e in estimation])
            # ax1.plot(_x, [x[1][0] for x in estimation])

            ax2 = fig.add_subplot(122)
            # ax2.plot([x[0] for x in estimation], [x[1][1] for x in estimation])
            ax2.plot(_x, [e[0][1] for e in estimation])
            ax2.plot(_x, [e[1][1] for e in estimation])

            plt.legend()
            plt.tight_layout()
            plt.show()
            self.assertTrue(True)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except RuntimeWarning as trw:
            logging.warning(trw)
            self.assertTrue(False)
        except RuntimeError as e:
            logging.error(e)
            self.assertTrue(False)


