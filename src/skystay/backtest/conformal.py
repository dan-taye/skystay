"""Split-conformal prediction intervals on log residuals.

Calibration residuals r = log(y / yhat) come from the out-of-fold (rolling
origin) predictions. Per market when it has at least `min_n` residuals,
otherwise pooled over the market's region (then over everything).
Two-sided intervals with the finite-sample correction:
    lo = r_(floor((n+1) a/2)),  hi = r_(ceil((n+1)(1 - a/2)))
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

LEVELS = (0.8, 0.95)


def _q(r: np.ndarray, level: float) -> tuple[float, float]:
    r = np.sort(r[np.isfinite(r)])
    n = len(r)
    a = 1 - level
    k_lo = max(math.floor((n + 1) * a / 2), 1)
    k_hi = min(math.ceil((n + 1) * (1 - a / 2)), n)
    return float(r[k_lo - 1]), float(r[k_hi - 1])


def calibrate(resid: pd.DataFrame, min_n: int = 30) -> dict:
    """resid: columns nationality, region, r. Returns {nationality: {level: [lo, hi], 'n': n, 'pool': ...}}."""
    out: dict = {}
    by_region = {k: g["r"].to_numpy() for k, g in resid.groupby("region")}
    everything = resid["r"].to_numpy()
    for (n, reg), g in resid.groupby(["nationality", "region"]):
        r = g["r"].to_numpy()
        if len(r) >= min_n:
            src, pool = r, "market"
        elif len(by_region.get(reg, [])) >= min_n:
            src, pool = by_region[reg], "region"
        else:
            src, pool = everything, "all"
        out[n] = {"n": int(len(src)), "pool": pool}
        for lv in LEVELS:
            out[n][str(lv)] = list(_q(src, lv))
    out["_all"] = {"n": int(len(everything)), "pool": "all", **{str(lv): list(_q(everything, lv)) for lv in LEVELS}}
    return out


def bands(q: dict, nationality: pd.Series, pred: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([q.get(n, q["_all"])[str(level)][0] for n in nationality])
    hi = np.array([q.get(n, q["_all"])[str(level)][1] for n in nationality])
    return pred * np.exp(lo), pred * np.exp(hi)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    ok = ~np.isnan(y)
    return float(((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])).mean()) if ok.any() else float("nan")
