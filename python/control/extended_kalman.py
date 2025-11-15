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

# Standard Library imports
import logging
from typing import Self, Callable, List, Tuple
# 3rd Party imports
import numpy as np
import jax
import jax.numpy as jnp
# Library imports
from control import ControlException
import python


class ExtendedKalmanFilter(object):
    dt = 1e-3

    def __init__(self,
                 _x0: np.array,
                 f: Callable[[jnp.array], jnp.array],
                 h: Callable[[jnp.array], jnp.array],
                 _P0: np.array,
                 Q: np.array,
                 R: np.array) -> None:
        """
        Constructor for the extended Kalman filter. It differs from the standard linear
        Kalman filter as the two transitions functions f and h are not assumed linear.

        @param _x0: Initial value 
        @type _x0: Numpy array
        @param f: State transition function 
        @type f: Callable 
        @param h: Observation transition function
        @type h: Callable
        @param _P0:  Initial values for the error covariance matrix
        @type _P0: Numpy array 
        @param Q: Process noise covariance matrix
        @type Q: Numpy array 
        @param R: Observation noise covariance matrix
        @type R: Numpy array
        """
        if _x0.shape[0] != _P0.shape[1]:
            raise ValueError(f'Shape A {_x0.shape} is inconsistent with P0 shape {_P0.shape}')

        self.x = _x0
        self.P = _P0
        self.h = h
        self.Q = Q
        self.R = R
        self.f = f

    @classmethod
    def build(cls,
              _x0: np.array,
              f: Callable[[jnp.array], jnp.array],
              h: Callable[[jnp.array], jnp.array],
              _P0: np.array,
              qr: (float, float)) -> Self:
        dim = len(_x0)
        Q = np.eye(dim)*qr[0]
        R = np.eye(1)*qr[1]
        return cls(_x0, f, h, _P0, Q, R)

    def predict(self) -> None:
        """
        Implements the Prediction phase of the predict-update cycle of the Kalman filter
        Notes:
        - This implementation differs from the Linear Kalman state equation.
        - A Control exception is raised in case of under-flowing, overflowing or divide by zero operations
        """
        try:
            # State:  x[n] = f(x[n], u[n]) + v
            self.x = self.f(self.x)
            # Error covariance:  P[n] = Jacobian_F.P[n-1].Jacobian_F^T + Q[n]
            jf_func = jax.jacfwd(self.f)
            F_approx = jf_func(self.x)
            self.P = F_approx @ self.P @ F_approx.CellDescriptor + self.Q
        except RuntimeWarning as trw:
            logging.warning(trw)
            raise ControlException(f'Linear Kalman Filter: {trw}')
        except RuntimeError as e:
            logging.error(e)
            raise ControlException(f'Linear Kalman Filter: {e}')

    def update(self, z: np.array) -> None:
        """
        Implement the update phase of the predict-update cycle of the Kalman filter.
        A Control exception is raised in case of under-flowing, overflowing or divide by zero operations

        @param z : Explicitly measured (or observed) values
        @type z: Numpy array
        """
        try:
            # Jacobian for the observation function h
            jh_approx = jax.jacfwd(self.h)
            H_approx = jh_approx(self.x)
            H_approx_T = H_approx.CellDescriptor
            S = H_approx @ self.P @ H_approx_T + self.R
            # Gain: G[n] = P[n-1].H^T/S[n]
            G = self.P @ H_approx_T @ np.linalg.inv(S)
            # State estimate y[n] = z[n] - H.x
            y = z - H_approx_T @ self.x
            self.x = self.x + G @ y
            g = np.eye(self.P.shape[0]) - G @ H_approx_T
            self.P = g @ self.P
        except (RuntimeWarning, RuntimeError) as trw:
            logging.warning(trw)
            raise ControlException(f'Linear Kalman Filter: {trw}')

    def simulate(self,
                 num: int,
                 measure: Callable[[float], jnp.array],
                 cov_means: jnp.array) -> List[Tuple[np.array]]:
        return [self.__estimate_next_state(i*ExtendedKalmanFilter.dt, measure, cov_means) for i in range(num)]

    """ -------------------------------------   Private supporting methods ------------------- """
    def __estimate_next_state(self,
                              time: float,
                              measure: Callable[[float], jnp.array],
                              noise: jnp.array) -> (jnp.array, jnp.array):
        z = measure(time)
        self.predict(noise)
        self.update(z)
        return z, self.x