import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from itertools import chain, combinations
from scipy.signal import savgol_filter, find_peaks
from sklearn.preprocessing import MinMaxScaler as Scaler

EPS = 1e-8


class PBFTPK:
    D: float
    F: float
    V_d: float
    k_a: float
    k_el: float
    tau: float
    alpha: float
    scaler: Scaler

    @staticmethod
    def _base_model(
        t: np.ndarray,
        D: float,
        F: float,
        V_d: float,
        k_a: float,
        k_el: float,
        tau: float,
    ) -> np.ndarray:
        X = np.zeros_like(t)

        def absorption_model(t: np.ndarray | float) -> np.ndarray | float:
            return (
                F
                * D
                * k_a
                / V_d
                / (k_a - k_el)
                * (np.exp(-k_el * t) - np.exp(-k_a * t))
            )

        X[t <= tau] = absorption_model(t[t <= tau])
        C_max = X[t <= tau][-1] if len(X[t <= tau]) > 0 else 1

        def elimination_model(t: np.ndarray | float) -> np.ndarray | float:
            return C_max * np.exp(-k_el * (t - tau))

        X[t > tau] = elimination_model(t[t > tau])

        return X

    def __init__(self, alpha: float = 1.0) -> None:
        self.initilized = False
        self.alpha = alpha

    def set_params(self, D: float, F: float, V_d: float, k_a: float, k_el: float):
        self.initilized = True
        self.D, self.F, self.V_d, self.k_a, self.k_el = D, F, V_d, k_a, k_el

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        self.initilized = True

        data = np.column_stack([t, X])
        self.scaler = Scaler()
        data = self.scaler.fit_transform(data)

        t = data[:, 0]
        X = data[:, 1]

        self.tau = t[np.argmax(X)]

        params_initial = [1.3, 0.5, 1, 1, 2]

        cons = LinearConstraint(
            [[0, 1, 0, 0, 0], [0, 0, 0, 1, -1]], [0.001, -np.inf], [1, 0.001]
        )

        def target_function(params):
            r = (PBFTPK._base_model(t, *params, tau=self.tau) - X) ** 2
            r[t > self.tau] *= self.alpha
            return np.mean(r)

        res = minimize(
            target_function, constraints=[
                cons], x0=params_initial, method="trust-constr"
        )

        self.D, self.F, self.V_d, self.k_a, self.k_el = res.x

    def sample(self, t: np.ndarray) -> np.ndarray:
        data = np.column_stack([t, np.zeros_like(t)])

        data = self.scaler.transform(data)

        t = data[:, 0]

        X = PBFTPK._base_model(
            t, self.D, self.F, self.V_d, self.k_a, self.k_el, self.tau
        )

        data = np.column_stack([t, X])

        data = self.scaler.inverse_transform(data)
        
        X = data[:, 1]

        return X

