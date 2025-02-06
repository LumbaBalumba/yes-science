from typing import Callable, Literal
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler as Scaler


class TauSeeker:
    method: Literal["minmax"] | Literal["peak"]

    def __init__(self, method: Literal["minmax"] | Literal["peak"]) -> None:
        self.method = method

    @staticmethod
    def peak_seek(t: np.ndarray, X: np.ndarray, t_max: float) -> tuple[float, float]:
        peaks, _ = find_peaks(X[t < t_max])
        if len(peaks) > 0:
            tau_2 = float(t[peaks[-1]])
        else:
            tau_2 = t[-1]
        volleys, _ = find_peaks(-X[t < tau_2])
        if len(volleys) > 0:
            tau_1 = float(t[volleys[-1]])
        else:
            tau_1 = t[0]
        return tau_1, tau_2

    @staticmethod
    def minmax_seek(t: np.ndarray, X: np.ndarray, t_max: float) -> tuple[float, float]:
        tau_2 = t[np.argmax(X[t < t_max])]
        tau_1 = t[np.argmax(X[t < tau_2])]
        return tau_1, tau_2

    def process(
        self, t: np.ndarray, X: np.ndarray, t_max: float
    ) -> tuple[float, float]:
        return (
            TauSeeker.minmax_seek(t, X, t_max)
            if self.method == "minmax"
            else TauSeeker.peak_seek(t, X, t_max)
        )


class BaseModel:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        pass

    def sample(self, t: np.ndarray) -> np.ndarray:
        return np.zeros_like(t)


class PBFTPK(BaseModel):
    D: float
    F: float
    V_d: float
    k_a: float
    k_el: float
    tau_1: float
    tau_2: float
    tau_3: float
    alpha: float
    scaler: Scaler
    base_model: Callable[
        [np.ndarray, float, float, float, float, float, float, float], np.ndarray
    ]
    tau_seeker: TauSeeker

    @staticmethod
    def PBFTPK0(
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
            t = np.array(t) - tau_1
            return F * D / V_d / k_el / (tau_2 - tau_1) * (1 - np.exp(-k_el * (t)))

        idx_a = (tau_1 < t) & (t <= tau_2)
        X[idx_a] = absorption_model(t[idx_a])
        C_max = X[idx_a][-1] if len(X[idx_a]) > 0 else 1

        def elimination_model(t: np.ndarray | float) -> np.ndarray | float:
            t = np.array(t) - tau_1
            return C_max * np.exp(-k_el * (t - tau_2))

        idx_el = t > tau_2
        X[idx_el] = elimination_model(t[idx_el])

        return X

    @staticmethod
    def PBFTPK1(
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
            t = np.array(t) - tau_1
            return (
                F
                * D
                * k_a
                / V_d
                / (k_a - k_el)
                * (np.exp(-k_el * (t)) - np.exp(-k_a * (t)))
            )

        idx_a = (tau_1 < t) & (t <= tau_2)
        X[idx_a] = absorption_model(t[idx_a])
        C_max = X[idx_a][-1] if len(X[idx_a]) > 0 else 1

        def elimination_model(t: np.ndarray | float) -> np.ndarray | float:
            t = np.array(t) - tau_1
            return C_max * np.exp(-k_el * (t - tau_2))

        idx_el = t > tau_2
        X[idx_el] = elimination_model(t[idx_el])

        return X

    def __init__(
        self,
        alpha: float = 1.0,
        tau_3: float = float("inf"),
        base_model: Literal["PBFTPK0"] | Literal["PBFTPK1"] = "PBFTPK1",
        tau_seek_method: Literal["minmax"] | Literal["peak"] = "peak",
    ) -> None:
        self.initilized = False
        self.alpha = alpha
        self.tau_3 = tau_3
        self.base_model = PBFTPK.PBFTPK0 if base_model == "PBFTPK0" else PBFTPK.PBFTPK1
        self.tau_seeker = TauSeeker(tau_seek_method)

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        self.initilized = True

        data = np.column_stack([t, X])
        self.scaler = Scaler()
        data = self.scaler.fit_transform(data)

        t = data[:, 0]
        X = data[:, 1]

        self.tau_1, self.tau_2 = self.tau_seeker.process(t, X, self.tau_3)

        X[t < self.tau_1] = 0.0

        params_initial = [1.3, 0.5, 1, 1, 2]

        cons = LinearConstraint([[0, 0, 0, 1, -1]], ub=-1e-4)

        bounds = Bounds(lb=1e-4, ub=[1e3, 1.0, 1e4, 300, 300])

        def target_function(params):
            D, F, V_d, k_a, k_el = params
            r = (
                self.base_model(t, D, F, V_d, k_a, k_el, self.tau_1, self.tau_2) - X
            ) ** 2
            r[t > self.tau_2] *= self.alpha
            return np.mean(r)

        res = minimize(
            target_function,
            constraints=cons,
            bounds=bounds,
            x0=params_initial,
            method="SLSQP",
        )

        self.D, self.F, self.V_d, self.k_a, self.k_el = res.x

    def sample(self, t: np.ndarray) -> np.ndarray:
        data = np.column_stack([t, np.zeros_like(t)])

        data = self.scaler.transform(data)

        t = data[:, 0]

        X = self.base_model(
            t, self.D, self.F, self.V_d, self.k_a, self.k_el, self.tau_1, self.tau_2
        )

        data = np.column_stack([t, X])

        data = self.scaler.inverse_transform(data)

        X = data[:, 1]

        X[X < 0] = 0

        return X


class EnsembledPBFTPK(BaseModel):
    n_models: int
    models: list[PBFTPK]
    base_model: Literal["PBFTPK0"] | Literal["PBFTPK1"]

    def __init__(
        self,
        n_models: int = 1,
        alpha: float | list[float] | np.ndarray = 1.0,
        base_model: Literal["PBFTPK0"] | Literal["PBFTPK1"] = "PBFTPK1",
    ):
        self.n_models = n_models
        self.alpha = np.ones(n_models) * alpha
        self.models = []
        self.base_model = base_model

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        r = X.copy()
        tau_3 = float("inf")
        for i in range(self.n_models):
            model = PBFTPK(alpha=self.alpha[i], tau_3=tau_3, base_model=self.base_model)
            model.fit(t, r)
            self.models.append(model)
            r -= self.models[i].sample(t)
            tau_3 = model.tau_2

    def sample(self, t: np.ndarray) -> np.ndarray:
        X = np.zeros_like(t)
        for model in self.models:
            X += model.sample(t)
        return X
