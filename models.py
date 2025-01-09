import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from sklearn.preprocessing import MinMaxScaler as Scaler


class PBFTPK:
    D: float
    F: float
    V_d: float
    k_a: float
    k_el: float
    tau_1: float
    tau_2: float
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
        tau_1: float,
        tau_2: float,
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

        idx_a = (tau_1 < t) & (t <= tau_2)
        X[idx_a] = absorption_model(t[idx_a])
        C_max = X[idx_a][-1] if len(X[idx_a]) > 0 else 1

        def elimination_model(t: np.ndarray | float) -> np.ndarray | float:
            return C_max * np.exp(-k_el * (t - tau_2))

        idx_el = t > tau_2
        X[idx_el] = elimination_model(t[idx_el])

        return X

    def __init__(self, alpha: float = 1.0) -> None:
        self.initilized = False
        self.alpha = alpha

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        self.initilized = True

        data = np.column_stack([t, X])
        self.scaler = Scaler()
        data = self.scaler.fit_transform(data)

        t = data[:, 0]
        X = data[:, 1]

        self.tau_1 = t[np.argmin(X)]
        X[t < self.tau_1] = 0.0
        self.tau_2 = t[np.argmax(X)]

        params_initial = [1.3, 0.5, 1, 1, 2]

        cons = LinearConstraint([[0, 0, 0, 1, -1]], -np.inf, 0.001)

        bounds = Bounds(
            lb=[1e-5, 1e-5, 1e-5, 1e-5, 1e-5], ub=[np.inf, 1.0, np.inf, np.inf, np.inf]
        )

        def target_function(params):
            r = (
                PBFTPK._base_model(t, *params, tau_1=self.tau_1, tau_2=self.tau_2) - X
            ) ** 2
            r[t > self.tau_2] *= self.alpha
            return np.mean(r)

        res = minimize(
            target_function,
            constraints=[cons],
            bounds=bounds,
            x0=params_initial,
            method="SLSQP",
        )

        self.D, self.F, self.V_d, self.k_a, self.k_el = res.x

    def sample(self, t: np.ndarray) -> np.ndarray:
        data = np.column_stack([t, np.zeros_like(t)])

        data = self.scaler.transform(data)

        t = data[:, 0]

        X = PBFTPK._base_model(
            t, self.D, self.F, self.V_d, self.k_a, self.k_el, self.tau_1, self.tau_2
        )

        data = np.column_stack([t, X])

        data = self.scaler.inverse_transform(data)

        X = data[:, 1]

        X[X < 0] = 0

        return X


class EnsembledPBFTPK:
    n_models: int
    models: list[PBFTPK]

    def __init__(self, n_models: int, alpha: float | list = 1.0):
        self.n_models = n_models
        self.models = []
        self.alpha = np.ones(n_models) * alpha

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        r = X.copy()
        for i in range(self.n_models):
            self.models.append(PBFTPK(alpha=self.alpha[i]))
            self.models[-1].fit(t, r)
            r -= self.models[-1].sample(t)

    def sample(self, t: np.ndarray) -> np.ndarray:
        X = np.zeros_like(t)
        for model in self.models:
            X += model.sample(t)
        return X


class FlexibleEnsembledPBFTPK:
    n_models: int
    model: EnsembledPBFTPK | None

    def __init__(self, n_models: int):
        assert n_models > 0
        self.n_models = n_models
        self.model = None

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        assert t.shape == X.shape
        bounds = Bounds(lb=[0] * self.n_models, ub=[10] * self.n_models)

        def target_function(alpha):
            model = EnsembledPBFTPK(self.n_models, alpha)
            model.fit(t, X)
            return np.mean((X - model.sample(t)) ** 2)

        alpha_initial = np.ones(self.n_models)

        res = minimize(
            target_function, bounds=[bounds], x0=alpha_initial, method="SLSQP"
        )

        self.model = EnsembledPBFTPK(self.n_models, res.x)

    def sample(self, t: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.sample(t)
