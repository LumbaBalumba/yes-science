import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class ExpModel:
    D: float
    F: float
    V_d: float
    k_a: float
    k_el: float
    tau: list
    modes: list

    @staticmethod
    def _base_model(
        t: np.ndarray,
        D: float,
        F: float,
        V_d: float,
        k_a: float,
        k_el: float,
        tau_arr: list[float],
        modes: list[str],
    ) -> np.ndarray:
        tau_arr += [np.inf]
        C_cur = 0.0
        tau_cur = 0.0

        X = np.zeros_like(t)

        def absorption_model(t: np.ndarray | float) -> np.ndarray | float:
            return C_cur + F * D * k_a / V_d / (k_a - k_el) * (
                np.exp(-k_el * t) - np.exp(-k_a * t)
            )

        def elimination_model(t: np.ndarray | float) -> np.ndarray | float:
            return C_cur * np.exp(-k_el * (t - tau_cur))

        for tau, mode in zip(tau_arr, modes):
            idx = (tau_cur < t) & (t <= tau)
            match mode:
                case "up":
                    X[idx] = absorption_model(t[idx])
                case "down":
                    X[idx] = elimination_model(t[idx])
            tau_cur = tau
            C_cur = X[idx][-1] if len(X[idx]) > 0 else C_cur
        return X

    def __init__(self, tau: list, modes: list) -> None:
        self.initilized = False
        self.tau = tau
        self.modes = modes

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        self.initilized = True

        params_initial = [1.3, 0.5, 1, 1, 2]

        cons = LinearConstraint(
            [[0, 1, 0, 0, 0], [0, 0, 0, 1, -1]], [0.001, -np.inf], [1, 0.001]
        )

        def target_function(params):
            r = (
                ExpModel._base_model(t, *params, tau_arr=self.tau, modes=self.modes) - X
            ) ** 2
            return np.mean(r)

        res = minimize(
            target_function, constraints=[cons], x0=params_initial, method="SLSQP"
        )

        self.D, self.F, self.V_d, self.k_a, self.k_el = res.x

    def sample(self, t=np.linspace(0, 100, 1000)):
        return ExpModel._base_model(
            t,
            D=self.D,
            F=self.F,
            V_d=self.V_d,
            k_a=self.k_a,
            k_el=self.k_el,
            tau_arr=self.tau,
            modes=self.modes,
        )


class TauModel:
    tau: list

    def __init__(self) -> None:
        pass

    def fit(self, t: np.ndarray, X: np.ndarray):
        X_smooth = gaussian_filter1d(X, sigma=2)

        peaks, _ = find_peaks(X_smooth)
        valleys, _ = find_peaks(-X_smooth)
        print(peaks, valleys)

        self.tau = list(t[sorted(peaks + valleys)])


class PBFTPK:
    exp_model: ExpModel
    tau_model: TauModel

    def __init__(self) -> None:
        pass

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        self.tau_model.fit(t, X)
