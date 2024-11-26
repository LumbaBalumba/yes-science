import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from itertools import chain, combinations
from scipy.signal import savgol_filter, find_peaks

EPS = 1e-8


class PBFTPK:
    D: float
    F: float
    V_d: float
    k_a: float
    k_el: float
    tau: float
    alpha: float

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

    def __init__(self, tau: float, alpha: float = 1.0) -> None:
        self.initilized = False
        self.tau = tau
        self.alpha = alpha

    def set_params(self, D: float, F: float, V_d: float, k_a: float, k_el: float):
        self.initilized = True
        self.D, self.F, self.V_d, self.k_a, self.k_el = D, F, V_d, k_a, k_el

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        self.initilized = True

        params_initial = [1.3, 0.5, 1, 1, 2]

        cons = LinearConstraint(
            [[0, 1, 0, 0, 0], [0, 0, 0, 1, -1]], [0.001, -np.inf], [1, 0.001]
        )

        def target_function(params):
            r = (PBFTPK._base_model(t, *params, tau=self.tau) - X) ** 2
            r[t > self.tau] *= self.alpha
            return np.mean(r)

        res = minimize(
            target_function, constraints=[cons], x0=params_initial, method="SLSQP"
        )

        self.D, self.F, self.V_d, self.k_a, self.k_el = res.x

    def sample(self, t: np.ndarray) -> np.ndarray:
        return PBFTPK._base_model(
            t, self.D, self.F, self.V_d, self.k_a, self.k_el, self.tau
        )


class MultiPBFTPK:
    N_PARAMS = 5

    n_models: int
    tau: list
    alpha: list

    def __init__(self, tau: list = [0.05], alpha: list = [1.0]) -> None:
        self.n_models = len(tau)
        self.params = np.random.uniform(
            low=1, high=2, size=(self.n_models, self.N_PARAMS)
        )
        self.tau = tau
        self.alpha = alpha

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:

        lb = [EPS, 0, EPS, EPS, EPS] * self.n_models
        ub = [np.inf, 1, np.inf, np.inf, np.inf] * self.n_models

        bounds = Bounds(lb, ub)

        lb = []
        ub = []
        matr = []

        for i in range(self.n_models):
            matr.append(
                [0] * self.N_PARAMS * i
                + [0, 0, 0, 1, -1]
                + [0] * self.N_PARAMS * (self.n_models - i - 1)
            )
            lb.append(-np.inf)
            ub.append(-EPS)

            if i > 0:
                row = [0] * self.N_PARAMS * self.n_models
                row[self.N_PARAMS - 1 + self.N_PARAMS * i] = 1
                row[self.N_PARAMS - 1 + self.N_PARAMS * (i - 1)] = -1
                matr.append(row)
                lb.append(EPS)
                ub.append(np.inf)

        cons = LinearConstraint(matr, lb, ub)

        def target_function(params):
            params = params.reshape((self.n_models, self.N_PARAMS))
            X_predict = np.zeros_like(X)
            for i, param_row in enumerate(params):
                D, F, V_d, k_a, k_el = param_row
                tau = self.tau[i]
                model = PBFTPK(tau, 1.0)
                model.set_params(D, F, V_d, k_a, k_el)
                X_predict += model.sample(t)
            r = (X - X_predict) ** 2
            for tau, alpha in zip(self.tau, self.alpha):
                r[t > tau] += r[t > tau] * (alpha - 1.0)
            return np.mean(r)

        res = minimize(
            target_function,
            method="COBYLA",
            x0=self.params.reshape((-1)),
            constraints=cons,
            bounds=bounds,
            options={"maxiter": 1000000},
        )
        if not res.success:
            print(res.message)

        self.params = res.x.reshape((self.n_models, self.N_PARAMS))

    def set_params(self, params):
        self.params = np.array(params)
        self.n_models = len(params)

    def sample(self, t: np.ndarray) -> np.ndarray:
        X_predict = np.zeros_like(t)
        for i, param_row in enumerate(self.params):
            tau = self.tau[i]
            D, F, V_d, k_a, k_el = param_row
            model = PBFTPK(tau, 1.0)
            model.set_params(D, F, V_d, k_a, k_el)
            X_predict += model.sample(t)
        return X_predict


class MainModel:
    model: MultiPBFTPK

    def __init__(self):
        pass

    @staticmethod
    def find_peaks(X: np.ndarray):
        return find_peaks(savgol_filter(X, 7, 3))[0]

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        def powerset(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

        X_peaks = MainModel.find_peaks(X)

        cur_peaks = []
        cur_result = np.inf

        for peaks in powerset(X_peaks):
            model = MultiPBFTPK(
                n_models=len(peaks), tau=list(peaks), alpha=[1] * len(peaks)
            )
            model.fit(t, X)

            r = np.mean((model.sample(t) - X) ** 2)
            if r < cur_result:
                cur_result = r
                cur_peaks = peaks
        self.model = MultiPBFTPK(
            n_models=len(cur_peaks), tau=list(cur_peaks), alpha=[1] * len(cur_peaks)
        )

    def sample(self, t: np.ndarray) -> np.ndarray:
        return self.model.sample(t)
