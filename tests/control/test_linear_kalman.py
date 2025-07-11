import unittest
import numpy as np
from control.linear_kalman import LinearKalmanFilter
from typing import NoReturn, List, AnyStr
from dataclasses import dataclass
from control import ControlException
import logging
import python

@dataclass
class KalmanPlot:
    estimated: List[np.array]
    observed: np.array
    title: AnyStr

class LinearKalmanFilterTest(unittest.TestCase):

    # @unittest.skip('Ignored')
    def test_init_2d(self):
        try:
            x0 = np.array([0.0, 0.1])
            P0 = np.eye(x0.shape[0])
            A = np.array([[1.0, 0.5], [0.0, 1.0]])
            H = np.array([[1.0, 0.0]])
            cov_means = (0.6, 1.0)
            kf = LinearKalmanFilter.build(x0, P0, A, H, cov_means)
            logging.info(str(kf))
            self.assertTrue(kf.A.shape[0] == 2)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except ControlException as e:
            logging.error(e)
            self.assertTrue(False)

    @unittest.skip('Ignored')
    def test_init_3d(self):
        try:
            x0 = np.array([0.0, 0.1, 0.1])
            P0 = np.eye(x0.shape[0])
            A = np.array([[1.0, 0.5, 1.0], [0.0, 1.0, 0.5], [1.0, 0.0, 1.0]])
            H = np.array([[1.0, 0.0, 0.0]])
            cov_means = (0.6, 1.0)
            kf = LinearKalmanFilter.build(x0, P0, A, H, cov_means)
            logging.info(str(kf))
            self.assertTrue(kf.A.shape[0] == 3)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except ControlException as _:
            self.assertTrue(False)

    @unittest.skip('Ignored')
    def test_simulate_2d(self):
        try:
            x0 = np.array([0.0, 0.1])
            P0 = np.eye(x0.shape[0])
            A = np.array([[1.0, 0.5], [0.0, 1.0]])
            H = np.array([[1.0, 0.0]])
            cov_means = (0.6, 1.0)
            num_points = 200
            rNoise = 3.4
            kf = LinearKalmanFilter.build(x0, P0, A, H, cov_means)
            estimation = kf.simulate(num_points, lambda i: np.array([i + np.random.normal(rNoise, rNoise)]),
                                     np.array([0.4, 0.6]))

            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(121)
            x: np.array = np.linspace(0, num_points, num_points)
            _x = x.tolist()
            ax1.plot(_x, [x[0] for x in estimation])
            ax1.plot(_x, [x[1][0] for x in estimation])

            ax2 = fig.add_subplot(122)
            # ax2.plot([x[0] for x in estimation], [x[1][1] for x in estimation])
            ax2.plot([x[1][1] for x in estimation], [x[0] for x in estimation])

            plt.legend()
            plt.tight_layout()
            plt.show()
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except ControlException as _:
            self.assertTrue(False)

    def test_simulate_trajectory(self):
        import math
        try:
            dt = 0.1
            ac = 0.5*dt*dt
            x0 = np.array([[0.0], [np.pi], [0.8], [0.2]])
            P0 = np.eye(x0.shape[0])
            A = np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
            H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            B = np.array([[ac, 0.0], [ac, 0.0], [dt, 0.0], [0.0, dt]])
            u = np.array([[1.0], [0.8]])
            Q_means = 0.6
            R_means = 1.0
            Q = np.eye(4) * Q_means
            R = np.eye(1) * R_means
            kf = LinearKalmanFilter(x0, P0, A, H, Q, R, u, B)
            num_points = 200

            def obs_generator_lin(i: int) -> np.array:
                return np.array([[i], [i]])

            def obs_generator_sqr(i: int) -> np.array:
                return np.array([[i], [i*i]])

            def obs_generator_exp(i: int) -> np.array:
                return np.array([[i], [math.exp(-i*0.1)]])

            def obs_generator_sqrt(i: int) -> np.array:
                return np.array([[i], [math.sqrt(20000.0*i)]])

            z_exp = [obs_generator_exp(i) for i in range(num_points)]
            z_lin = [obs_generator_lin(i) for i in range(num_points)]
            z_sqr = [obs_generator_sqr(i) for i in range(num_points)]
            z_sqrt = [obs_generator_sqrt(i) for i in range(num_points)]

            estimation1 = kf.simulate(num_points,
                                    lambda i: obs_generator_lin(i),
                                    np.array([[0.4], [0.6], [0.1], [0.2]]))
            kalman_plot1 = KalmanPlot(estimation1, z_lin, f'x=n, y=n')

            estimation2 = kf.simulate(num_points,
                                    lambda i: obs_generator_sqr(i),
                                    np.array([[0.4], [0.6], [0.1], [0.2]]))
            kalman_plot2 = KalmanPlot(estimation2, z_sqr, f'x=n, y=n*n')
            LinearKalmanFilterTest.__plot_observed_estimate(num_points, kalman_plot1, kalman_plot2)

            estimation3 = kf.simulate(num_points,
                                     lambda i: obs_generator_exp(i),
                                     np.array([[0.4], [0.6], [0.1], [0.2]]))
            kalman_plot3 = KalmanPlot(estimation3, z_exp, f'x=n, y=exp(-0.1.n)')

            estimation4 = kf.simulate(num_points,
                                     lambda i: obs_generator_sqrt(i),
                                     np.array([[0.4], [0.6], [0.1], [0.2]]))
            kalman_plot4 = KalmanPlot(estimation4, z_sqrt, f'x=n, y=sqrt(20000.n)')
            LinearKalmanFilterTest.__plot_observed_estimate(num_points, kalman_plot3, kalman_plot4)
            self.assertTrue(True)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except ControlException as _:
            self.assertTrue(False)

    @unittest.skip('Ignored')
    def test_simulate_3d(self):
        try:
            x0 = np.array([0.0, 0.1, 0.1])
            P0 = np.eye(x0.shape[0])
            A = np.array([[1.0, 0.5, 1.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
            H = np.array([[1.0, 0.2, 0.0]])
            cov_means = (0.6, 1.0)

            kf = LinearKalmanFilter.build(x0, P0, A, H, cov_means)

            num_points = 200
            rNoise = 3.4
            estimation = kf.simulate(num_points,
                                     lambda i: np.array([i + np.random.normal(rNoise, rNoise)]),
                                     np.array([0.4, 0.6, 0.1]))
            LinearKalmanFilterTest.__plot_observed_estimate(num_points, estimation)
        except AssertionError as e:
            logging.error(e)
            self.assertTrue(False)
        except ControlException as _:
            self.assertTrue(False)

    """ --------------  Private Supporting Methods ---------------------------- """

    @staticmethod
    def __plot_observed_estimate(
            num_points: int,
            kalman_plot1: KalmanPlot,
            kalman_plot2: KalmanPlot) -> None:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(8, 6))

        x: np.array = np.linspace(0, num_points, num_points)
        _x = x.tolist()
        axs[0].plot(_x, [x[0] for x in kalman_plot1.estimated])
        axs[0].plot(_x, [x[0] for x in kalman_plot1.observed])
        axs[0].set_title(kalman_plot1.title, fontdict={'fontsize': 15, 'fontweight': 'bold', 'family': 'serif'})
        axs[0].set_xlabel('X input', fontdict={'fontsize': 12, 'fontweight': 'regular', 'family': 'serif'})
        axs[0].set_ylabel('Trajectory', fontdict={'fontsize': 12, 'fontweight': 'regular', 'family': 'serif'})

        axs[1].plot(_x, [x[0] for x in kalman_plot2.estimated], color='red', label='estimated')
        axs[1].plot(_x, [x[0] for x in kalman_plot2.observed], color='blue', label='observed')
        axs[1].set_title(kalman_plot2.title, fontdict={'fontsize': 15, 'fontweight': 'bold', 'family': 'serif'})
        axs[1].set_xlabel('X input', fontdict={'fontsize': 12, 'fontweight': 'regular', 'family': 'serif'})
        axs[1].set_ylabel('Trajectory', fontdict={'fontsize': 12, 'fontweight': 'regular', 'family': 'serif'})

        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def __plot_trajectories(kalman_plots: List[KalmanPlot]) -> NoReturn:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs[0, 0].scatter(
            [x[0] for x in kalman_plots[0].estimated],
            [x[1] for x in kalman_plots[0].estimated],
            color='red',
            label="estimated")
        axs[0, 0].scatter(
            [z[0] for z in kalman_plots[0].observed],
            [z[1] for z in kalman_plots[0].observed],
            color='blue',
            label='measured')
        axs[0, 0].set_title(kalman_plots[0].title, fontdict={'fontsize': 15, 'fontweight': 'bold', 'family': 'serif'})
        axs[0, 0].set_xlabel('X trajectory', fontdict={'fontsize': 12, 'fontweight': 'regular', 'family': 'serif'})
        axs[0, 0].set_ylabel('Y trajectory', fontdict={'fontsize': 12, 'fontweight': 'regular', 'family': 'serif'})

        axs[0, 1].scatter(
            [x[0] for x in kalman_plots[1].estimated],
            [x[1] for x in kalman_plots[1].estimated],
            color='red')
        axs[0, 1].scatter(
            [z[0] for z in kalman_plots[1].observed],
            [z[1] for z in kalman_plots[1].observed],
            color='blue')
        axs[0, 1].set_title(kalman_plots[1].title, fontdict={'fontsize': 15, 'fontweight': 'bold', 'family': 'serif'})
        axs[0, 1].set_xlabel('X trajectory', fontdict={'fontsize': 12, 'fontweight': 'regular', 'family': 'serif'})
        axs[0, 1].set_ylabel('Y trajectory', fontdict={'fontsize': 12, 'fontweight': 'regular', 'family': 'serif'})

        axs[1, 0].scatter(
            [x[0] for x in kalman_plots[2].estimated],
            [x[1] for x in kalman_plots[2].estimated],
            color='red')
        axs[1, 0].scatter(
            [z[0] for z in kalman_plots[2].observed],
            [z[1] for z in kalman_plots[2].observed],
            color='blue')
        axs[1, 0].set_title(kalman_plots[2].title, fontdict={'fontsize': 15, 'fontweight': 'bold', 'family': 'serif'})
        axs[1, 0].set_xlabel('X trajectory', fontdict={'fontsize': 12, 'fontweight': 'regular', 'family': 'serif'})
        axs[1, 0].set_ylabel('Y trajectory', fontdict={'fontsize': 12, 'fontweight': 'regular', 'family': 'serif'})

        axs[1, 1].scatter(
            [x[0] for x in kalman_plots[3].estimated],
            [x[1] for x in kalman_plots[3].estimated],
            color='red',
            label='estimated')
        axs[1, 1].scatter(
            [z[0] for z in kalman_plots[3].observed],
            [z[1] for z in kalman_plots[3].observed],
            color='blue',
            label='measured')
        axs[1, 1].set_title(kalman_plots[3].title, fontdict={'fontsize': 15, 'fontweight': 'bold', 'family': 'serif'})
        axs[1, 1].set_xlabel('X trajectory', fontdict={'fontsize': 12, 'fontweight': 'regular', 'family': 'serif'})
        axs[1, 1].set_ylabel('Y trajectory', fontdict={'fontsize': 12, 'fontweight': 'regular', 'family': 'serif'})

        plt.legend()
        plt.tight_layout()
        plt.show()
