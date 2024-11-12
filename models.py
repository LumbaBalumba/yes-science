from typing import Callable
import numpy as np
from scipy.optimize import curve_fit, LinearConstraint, minimize


class PBFTPKModel:
    D: float
    F: float
    V_d: float
    k_a: float
    k_el: float
    tau: float

    @staticmethod
    def _base_model(
        t: np.ndarray,
        D: float,
        F: float,
        V_d: float,
        k_a: float,
        k_el: float,
        tau: float,
        der: bool = False
    ) -> np.ndarray:
        t_a = t[t <= tau]
        t_el = t[t > tau]

        def absorption_model(t: np.ndarray | float) -> np.ndarray | float:
            return (
                F * D * k_a / V_d / (k_a - k_el) *
                (np.exp(-k_el * t) - np.exp(-k_a * t))
            )

        C_a = absorption_model(t_a)

        def elimination_model(t: np.ndarray | float) -> np.ndarray | float:
            return absorption_model(tau) * np.exp(-k_el * (t - tau))

        C_el = elimination_model(t_el)

        return np.concatenate([C_a, C_el])

    def __init__(self) -> None:
        self.initilized = False

    def fit(self, t: np.ndarray, x: np.ndarray) -> None:
        self.initilized = True
        self.tau = t[np.argmax(x)]

        x0 = [1.3, 0.5, 1, 1, 2]

        cons = LinearConstraint(
            [[0, 1, 0, 0, 0], [0, 0, 0, 1, -1]], [0.001, -np.inf], [1, 0.001])

        res = minimize(lambda params: np.mean(np.abs(
            PBFTPKModel._base_model(
                t, *params, tau=self.tau) - x
        )),
            constraints=[cons],
            x0=x0,
            method='trust-constr'
            )

        self.D, self.F, self.V_d, self.k_a, self.k_el = res.x

    def sample(self, t=np.linspace(0, 100, 1000)):
        return PBFTPKModel._base_model(t, D=self.D, F=self.F, V_d=self.V_d, k_a=self.k_a, k_el=self.k_el, tau=self.tau)
