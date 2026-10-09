"""Error metrics. WMAPE is primary."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ok = ~np.isnan(y) & ~np.isnan(yhat)
    return y[ok], yhat[ok]


def wmape(y, yhat) -> float:
    """Sum |y - yhat| / sum |y|."""
    y, yhat = _clean(y, yhat)
    den = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / den) if den > 0 else float("nan")


def bias(y, yhat) -> float:
    """Sum (yhat - y) / sum y (positive = over-prediction)."""
    y, yhat = _clean(y, yhat)
    den = y.sum()
    return float((yhat - y).sum() / den) if den > 0 else float("nan")


def mase(df: pd.DataFrame, ycol: str, pcol: str, scale: pd.Series) -> float:
    """Mean over markets of MAE / in-sample seasonal-naive MAE (scale indexed by nationality)."""
    vals = []
    for n, g in df.groupby("nationality"):
        sc = scale.get(n, np.nan)
        y, yhat = _clean(g[ycol], g[pcol])
        if len(y) and sc and np.isfinite(sc) and sc > 0:
            vals.append(np.abs(y - yhat).mean() / sc)
    return float(np.mean(vals)) if vals else float("nan")


def snaive_scale(history: pd.DataFrame, col: str) -> pd.Series:
    """In-sample MAE of the seasonal-naive forecast per nationality (MASE denominator)."""
    h = history[["nationality", "month", col]].dropna().copy()
    lag = h.copy()
    lag["month"] = lag["month"] + pd.DateOffset(months=12)
    m = h.merge(lag, on=["nationality", "month"], suffixes=("", "_lag"))
    return (m[col] - m[f"{col}_lag"]).abs().groupby(m["nationality"]).mean()


def block(df: pd.DataFrame, ycol: str, pcol: str, scale: pd.Series | None = None) -> dict:
    out = {"wmape": wmape(df[ycol], df[pcol]), "bias": bias(df[ycol], df[pcol]), "n": int(df[ycol].notna().sum()),
           "actual_total": float(df[ycol].sum())}
    if scale is not None:
        out["mase"] = mase(df, ycol, pcol, scale)
    return out


def grouped(df: pd.DataFrame, by: str, ycol: str, pcol: str) -> dict:
    return {str(k): {"wmape": wmape(g[ycol], g[pcol]), "bias": bias(g[ycol], g[pcol]),
                     "actual_total": float(g[ycol].sum())}
            for k, g in df.groupby(by)}
