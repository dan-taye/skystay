"""Seasonal-naive baseline: same month last year."""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict(history: pd.DataFrame, rows: pd.DataFrame, col: str = "guests") -> np.ndarray:
    """Value 12 months earlier from `history` (nationality, month, col); falls back to the
    same-month-of-year mean of the history when last year's value is missing."""
    h = history[["nationality", "month", col]].dropna()
    lag = rows[["nationality", "month"]].copy()
    lag["src"] = lag["month"] - pd.DateOffset(months=12)
    out = lag.merge(h.rename(columns={"month": "src"}), on=["nationality", "src"], how="left")[col].to_numpy()
    if np.isnan(out).any():
        moy_mean = h.assign(moy=h["month"].dt.month).groupby(["nationality", "moy"])[col].mean()
        keys = list(zip(rows["nationality"], rows["month"].dt.month, strict=True))
        fill = np.array([moy_mean.get(k, np.nan) for k in keys])
        out = np.where(np.isnan(out), fill, out)
    return out
