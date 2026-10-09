"""Rolling-origin folds (expanding window, 12-month horizon) and the final holdout."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    cutoff: pd.Timestamp  # first test month; training uses months strictly before it
    test: pd.DatetimeIndex
    holdout: bool
    train_months: int

    def describe(self) -> dict:
        return {
            "name": self.name,
            "cutoff": self.cutoff.strftime("%Y-%m"),
            "test_first": self.test[0].strftime("%Y-%m"),
            "test_last": self.test[-1].strftime("%Y-%m"),
            "train_months": self.train_months,
            "holdout": self.holdout,
        }


def make_folds(months: pd.DatetimeIndex, n_folds: int = 3, horizon: int = 12, holdout: int = 12,
               min_train: int = 24) -> list[Fold]:
    months = pd.DatetimeIndex(sorted(months))
    if len(months) < holdout + horizon + min_train:
        raise ValueError(f"only {len(months)} months: need >= {holdout + horizon + min_train} for one fold + holdout")
    folds = []
    h_start = len(months) - holdout
    for k in range(n_folds, 0, -1):
        start = h_start - k * horizon
        if start < min_train:
            continue
        folds.append(Fold(f"fold{len(folds) + 1}", months[start], months[start:start + horizon], False, start))
    folds.append(Fold("holdout", months[h_start], months[h_start:], True, h_start))
    return folds
