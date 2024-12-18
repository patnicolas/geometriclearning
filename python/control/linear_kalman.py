__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import numpy as np
from typing import Self, NoReturn, Callable, List, Tuple, AnyStr

"""
Numpy implementation of the Kalman estimator (filter) for linear or pseudo-linear processes.
There are two constructors
__init__:  Default constructor with all the components required for the Kalman filter equations fully
             defined
build: Alternative or simplified constructor for the Kalman filter with non control and null covariance 
       values for the Process and Measurement noises
"""


class LinearKalmanFilter(object):

    def __init__(self,
                 _x0: np.array,
                 _P0: np.array,
                 _A: np.array,
                 _H: np.array,
                 _Q: np.array,
                 _R: np.array,
                 _u0: np.array = None,
                 _B: np.array = None) -> None:
        """
        Default and fully specified constructor for the standard, linear Kalman filter
        Parameters:
        _x0 : Initial values for the estimated state
        _P0 : Initial values for the error covariance matrix
        _A : State transition matrix (from state x[n-1] to state x[n])
        _H : States to observations (or measurements) matrix
        _Q : Process noise covariance matrix
        _R : Observation or measurement matrix
        _u0 : Optional initial value of the control variables
        _B :  Optional control matrix (No control if None)
        """
        self.x = _x0
        self.P = _P0
        self.A = _A
        self.H = _H
        self.Q = _Q
        self.R = _R
        self.u = _u0
        self.B = _B

    @classmethod
    def build(cls, _x0: np.array, _P0: np.array, _A: np.array, _H: np.array, qr: (float, float)) -> Self:
        """
        Alternative constructor for the simplified Kalman filter:
        - No control input
        - Process and Measurement noise matrices have non-diagonal elements null (variance only)

        Parameters
        _x0 :  Initial values for the estimated state
        _P0 : Initial values for the error covariance matrix
        _A : State transition matrix (from state x[n-1] to state x[n])
        _H : States to observations (or measurements) matrix
        qr : Tuple for the mean values (variance) of Process and measurement noises

        Returns Instance of KalmanFilter
        """
        dim = len(_x0)
        Q = np.eye(dim)*qr[0]
        R = np.eye(1)*qr[1]
        return cls(_x0, _P0, _A, _H, Q, R)

    def __str__(self) -> AnyStr:
        return f'\nA state transition:\n{self.A}\nH observations:\n{self.H}\nP covariance:{self.P}\nQ:\n{self.Q}\nR:\n{self.R}\nx state:\n{self.x}'

    def predict(self, v: np.array) -> NoReturn:
        """
        Implements the Prediction phase of the predict-update cycle of the Kalman filter
        Parameters
        v : Noise for the process
        """
        # State:  x[n] = A.x~[n-1] + B.u[n-1] + v
        self.x = self.A @ self.x + v if self.B is None else self.A @ self.x + self.B @ self.u + v
        # Error covariance:  P[n] = A[n].P[n-1].A[n]^T + Q[n]
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z: np.array) -> NoReturn:
        """
        Implement the update phase of the predict-update cycle of the Kalman filter. Each equation
        is commented
        Parameters:
        z : Explicitly measured (or observed) values
        """
        # Innovation:  S[n] = H.P[n-1].H^T + R[n]
        S = self.H @ self.P @ self.H.T + self.R
        # Gain: G[n] = P[n-1].H^T/S[n]
        G = self.P @ self.H.T @ np.linalg.inv(S)

        # State estimate y[n] = z[n] - H.x
        y = z - self.H @ self.x
        self.x = self.x + G @ y
        g = np.eye(self.P.shape[0]) - G @ self.H
        self.P = g @ self.P


    def simulate(self,
                 num_measurements: int,
                 measure: Callable[[int], np.array],
                 cov_means: np.array) -> List[np.array]:
        """
        Simulate the execution of Kalman filter using a synthetic set of num_measurements sobservations
        generated by a function, measure
        Parameters:
        num_measurements :  Number of observations or measurements
        measure : Function or lambda that generate observation data at any given step s => measurement
        cov_means : Means for the noise covariance matrix

        Returns:
        List of estimated data points
        """
        return [self.__estimate_next_state(i, measure, cov_means) for i in range(num_measurements)]

    """ -------------------------------------   Private supporting methods ------------------- """
    def __estimate_next_state(self, state_index: int, observed: Callable[[int], np.array], noise: np.array) -> np.array:
        z = observed(state_index)
        self.predict(noise)
        self.update(z)
        return self.x

