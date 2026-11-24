"""Shared helpers of auswertung_tuned.py and auswertung_kandidaten.py (formula fit of 17.11.)."""
import math

import numpy as np
from scipy.optimize import minimize


def fit_total(c):
    x = np.log(c["ratio"])

    def pred(q):
        return np.exp(q[0] + q[1] * x) + np.exp(q[2] + q[3] * x) / c["n"]

    obj = lambda q: float(np.sum(c["mse"] / pred(q) + np.log(pred(q))))
    opts = {"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000, "maxfev": 20000}
    best = min((minimize(obj, x0=s, method="Nelder-Mead", options=opts) for s in ([2.5, -1.24, 8.0, -1.0], [2.0, -1.0, 8.5, -0.5])),
               key=lambda r: r.fun)
    q = best.x
    return dict(b=math.exp(q[0] / 2), be=q[1] / 2, v=math.exp(q[2]), ve=q[3])


def cells(q):
    return q.groupby(["design", "n"]).agg(mse=("err", lambda e: float((e ** 2).mean())), ratio=("ratio", "first")).reset_index()


def needed(par, r, m, k):
    room = (m / k) ** 2 - (par["b"] * r ** par["be"]) ** 2
    return None if room <= 0 else math.ceil(par["v"] * r ** par["ve"] / room - 1e-9)
