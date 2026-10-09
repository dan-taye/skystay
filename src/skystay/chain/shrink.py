"""Empirical-Bayes shrinkage of month-of-year estimates toward an annual mean."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z80 = 1.2816


@dataclass
class MoyEstimate:
    value: np.ndarray  # (12,) shrunk estimate per month of year
    se: np.ndarray  # (12,) posterior standard error of the estimate
    n: np.ndarray  # (12,) observations per month of year
    annual: float
    annual_se: float
    n_total: int
    resid_sd: float  # month-to-month variability around the moy mean

    def interval80(self) -> tuple[np.ndarray, np.ndarray]:
        return self.value - Z80 * self.se, self.value + Z80 * self.se


def eb_moy(
    values: np.ndarray,
    moy: np.ndarray,
    prior: np.ndarray | float | None = None,
    prior_strength: float = 6.0,
    fallback_sd: float | None = None,
) -> MoyEstimate:
    """Shrink per-month-of-year means toward the series' annual mean.

    Thin series (fewer than 12 observations) also have their annual mean shrunk
    toward `prior` (e.g. the region mean) with weight `prior_strength`
    pseudo-observations. Months with no data take the (shrunk) annual mean.
    """
    values = np.asarray(values, dtype=float)
    moy = np.asarray(moy, dtype=int)
    ok = ~np.isnan(values)
    v, m = values[ok], moy[ok]
    n_tot = int(ok.sum())
    prior_arr = None if prior is None else np.broadcast_to(np.asarray(prior, dtype=float), (12,))
    if n_tot == 0:
        if prior_arr is None:
            nan = np.full(12, np.nan)
            return MoyEstimate(nan, nan, np.zeros(12), np.nan, np.nan, 0, np.nan)
        sd = fallback_sd if fallback_sd is not None else 0.1 * float(np.nanmean(np.abs(prior_arr)))
        return MoyEstimate(prior_arr.copy(), np.full(12, sd), np.zeros(12), float(prior_arr.mean()), sd, 0, sd)

    n = np.array([(m == k).sum() for k in range(1, 13)], dtype=float)
    means = np.array([v[m == k].mean() if n[k - 1] else np.nan for k in range(1, 13)])
    resid = v - means[m - 1]
    dof = max(n_tot - int((n > 0).sum()), 1)
    sigma2 = float((resid**2).sum() / dof) if n_tot > (n > 0).sum() else float(np.var(v)) or 1e-6
    sigma2 = max(sigma2, 1e-10)
    grand = float(v.mean())
    if prior_arr is not None and n_tot < 12:
        grand = (n_tot * grand + prior_strength * float(prior_arr.mean())) / (n_tot + prior_strength)
    has = n > 0
    if has.sum() >= 3:
        between = float(np.var(means[has], ddof=1)) - float(np.mean(sigma2 / n[has]))
    else:
        between = 0.0
    tau2 = max(between, 1e-10)
    out = np.full(12, grand)
    se = np.full(12, np.sqrt(sigma2 / max(n_tot, 1) + tau2))
    w = np.zeros(12)
    w[has] = tau2 / (tau2 + sigma2 / n[has])
    out[has] = w[has] * means[has] + (1 - w[has]) * grand
    se[has] = np.sqrt(1.0 / (1.0 / tau2 + n[has] / sigma2))
    annual_se = float(np.sqrt(sigma2 / n_tot))
    return MoyEstimate(out, se, n, grand, annual_se, n_tot, float(np.sqrt(sigma2)))
