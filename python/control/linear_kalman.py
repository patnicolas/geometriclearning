__author__ = "Patrick Nicolas"
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

import numpy as np
from typing import Self, Callable, List, Tuple, AnyStr
from control import ControlException
import logging
import python


class LinearKalmanFilter(object):
    """
    Numpy implementation of the Kalman estimator (filter) for linear or pseudo-linear processes.
    There are two constructors
    __init__:  Default constructor with all the components required for the Kalman filter equations fully
                 defined
    build: Alternative or simplified constructor for the Kalman filter with non control and null covariance
           values for the Process and Measurement noises
    """

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

        @param _x0 : Initial values for the estimated state
        @type _x0: Numpy array
        @param _P0 : Initial values for the error covariance matrix
        @type _P0: Numpy array
        @param _A : State transition matrix (from state x[n-1] to state x[n])
        @type _A: Numpy array
        @param _H : States to observations (or measurements) matrix
        @type _H: Numpy array
        @param _Q : Process noise covariance matrix
        @type _Q: Numpy array
        @param _R : Observation or measurement matrix
        @type _R: Numpy array
        @param _u0 : Optional initial value of the control variables
        @type _u0: Numpy array
        @param _B :  Optional control matrix (No control if None)
        @type _B: Numpy array
        """
        LinearKalmanFilter.__validate(_x0, _H, _A, _P0)

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

        @param _x0 :  Initial values for the estimated state
        @type _x0: Numpy array
        @param _P0 : Initial values for the error covariance matrix
        @type _P0: Numpy array
        @param  _A : State transition matrix (from state x[n-1] to state x[n])
        @type _A: Numpy array
        @param _H : States to observations (or measurements) matrix
        @type _A: Numpy array
        @param qr : Tuple for the mean values (variance) of Process and measurement noises
        @type qr: Tuple[numpy.array, numpy.array]
        @return Instance of KalmanFilter
        """
        dim = len(_x0)
        Q = np.eye(dim)*qr[0]
        R = np.eye(1)*qr[1]
        return cls(_x0, _P0, _A, _H, Q, R)

    def __str__(self) -> AnyStr:
        return f'\nA state transition:\n{self.A}\nH observations:\n{self.H}\nP covariance:{self.P}\nQ:\n{self.Q}\nR:\n{self.R}\nx state:\n{self.x}'

    def predict(self, v: np.array) -> None:
        """
        Implements the Prediction phase of the predict-update cycle of the Kalman filter. A Control exception
        is raised in case of under-flowing, overflowing or divide by zero operations.

        @param v: Noise for the process
        @type v: Numpy array
        """
        try:
            # State:  x[n] = A.x~[n-1] + B.u[n-1] + v
            self.x = self.A @ self.x + v if self.B is None else self.A @ self.x + self.B @ self.u + v
            # Error covariance:  P[n] = A[n].P[n-1].A[n]^T + Q[n]
            self.P = self.A @ self.P @ self.A.T + self.Q
        except RuntimeWarning as trw:
            logging.warning(trw)
            raise ControlException(f'Linear Kalman Filter: {trw}')
        except RuntimeError as e:
            logging.error(e)
            raise ControlException(f'Linear Kalman Filter: {e}')

    def update(self, z: np.array) -> None:
        """
        Implement the update phase of the predict-update cycle of the Kalman filter. Each equation
        is commented. A Control exception is raised in case of under-flowing, overflowing or divide by zero operation

        @param z : Explicitly measured (or observed) values
        @type z: Numpy array
        """
        try:
            # Innovation:  S[n] = H.P[n-1].H^T + R[n]
            S = self.H @ self.P @ self.H.T + self.R
            # Gain: G[n] = P[n-1].H^T/S[n]
            G = self.P @ self.H.T @ np.linalg.inv(S)

            # State estimate y[n] = z[n] - H.x
            y = z - self.H @ self.x
            self.x = self.x + G @ y
            g = np.eye(self.P.shape[0]) - G @ self.H
            self.P = g @ self.P
        except RuntimeWarning as trw:
            logging.warning(trw)
            raise ControlException(f'Linear Kalman Filter: {trw}')
        except RuntimeError as e:
            logging.error(e)
            raise ControlException(f'Linear Kalman Filter: {e}')

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

    @staticmethod
    def __validate(_x0: np.array,
                   _P0: np.array,
                   _A: np.array,
                   _H: np.array) -> None:
        assert _A.shape[0] == _x0.shape[0], \
                f'Shape A {_A.shape} is inconsistent with x0 shape {_x0.shape}'
        assert _A.shape[0] == _A.shape[1], f'A shape {_A.shape} should be square'
        assert _A.shape[0] == _H.shape[0], \
                f'Shape A {_A.shape} is inconsistent with H shape {_H.shape}'
        assert _A.shape[0] == _P0.shape[1], \
                f'Shape A {_A.shape} is inconsistent with P0 shape {_P0.shape}'


