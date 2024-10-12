__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2024  All rights reserved."

import numpy as np
import jax
import jax.numpy as jnp
from typing import Self, NoReturn, Callable, List, Tuple, Any


class ExtendedKalmanFilter(object):
    dt = 1e-3
    def __init__(self,
                 _x0: np.array,
                 f: Callable[[jnp.array], jnp.array],
                 h: Callable[[jnp.array], jnp.array],
                 _P0: np.array,
                 Q: np.array,
                 R: np.array) -> None:
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

    def predict(self, u: np.array = 0.0) -> NoReturn :
        # State:  x[n] = f(x[n], u[n]) + v
        # self.x = self.f(jnp.array([self.x, u])) if u != 0.0 else self.f(self.x)
        self.x = self.f(self.x)
        # Error covariance:  P[n] = Jacobian_F.P[n-1].Jacobian_F^T + Q[n]
        jf_func = jax.jacfwd(self.f)
        F_approx = jf_func(self.x)
        self.P = F_approx @ self.P @ F_approx.T + self.Q

    def update(self, z: np.array) -> NoReturn:
        # Jacobian for the observation function h
        jh_approx = jax.jacfwd(self.h)
        H_approx = jh_approx(self.x)
        H_approx_T = H_approx.T
        S = H_approx @ self.P @ H_approx_T + self.R
        # Gain: G[n] = P[n-1].H^T/S[n]
        G = self.P @ H_approx_T @ np.linalg.inv(S)
        # State estimate y[n] = z[n] - H.x
        y = z - H_approx_T @ self.x
        self.x = self.x + G @ y
        g = np.eye(self.P.shape[0]) - G @ H_approx_T
        self.P = g @ self.P

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
