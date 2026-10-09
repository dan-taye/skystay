"""Layer 3: LightGBM conversion layer with monotone constraints.

Two targets, chosen by back-test:
- "ml_only": log(1 + G)                      -> G = exp(f) - 1
- "hybrid":  log(G / Layer-1 prediction)     -> G = Layer1 * exp(f)

The air-visitor feature is constrained monotone increasing, so predicted
guests never fall when terminating seats or visitors rise (Layer 1 is itself
increasing in seats). Market capacity share is deliberately NOT a feature: it
would let a route change move every other market's prediction, breaking the
bridge-locality guarantee (see docs/DECISIONS.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURES = ["nat_code", "moy", "holiday_days", "event_days", "trend", "log_air", "lag12"]
MONOTONE = [0, 0, 0, 0, 0, 1, 0]
PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 8,
    "max_depth": 3,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "monotone_constraints_method": "advanced",
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
}


@dataclass
class GbmModel:
    booster: lgb.Booster
    target: str
    nat_codes: dict[str, int]
    best_iter: int
    val_months: list[str]


def _X(df: pd.DataFrame, nat_codes: dict[str, int]) -> pd.DataFrame:
    X = df.copy()
    X["nat_code"] = X["nationality"].map(nat_codes).fillna(-1).astype(int)
    return X[FEATURES].astype(float)


def _y(df: pd.DataFrame, target: str) -> np.ndarray:
    if target == "ml_only":
        return np.log1p(df["guests"].to_numpy(dtype=float))
    return np.log(df["guests"].to_numpy(dtype=float) / np.maximum(df["l1_guests"].to_numpy(dtype=float), 1.0))


def fit(train: pd.DataFrame, target: str = "hybrid", val_months: int = 12, max_rounds: int = 800) -> GbmModel:
    if target not in ("ml_only", "hybrid"):
        raise ValueError(target)
    d = train[(train["guests"] > 0)].copy()
    if target == "hybrid":
        d = d[d["l1_guests"] > 0]
    nat_codes = {n: i for i, n in enumerate(sorted(d["nationality"].unique()))}
    w_nat = d.groupby("nationality")["guests"].transform("mean")
    w = (w_nat / w_nat.mean()).to_numpy()
    months = sorted(d["month"].unique())
    cut = months[-val_months] if len(months) > val_months + 12 else None
    params = dict(PARAMS, monotone_constraints=MONOTONE)
    cat = ["nat_code"]
    if cut is not None:
        tr, va = d["month"] < cut, d["month"] >= cut
        dtr = lgb.Dataset(_X(d[tr], nat_codes), _y(d[tr], target), weight=w[tr.to_numpy()], categorical_feature=cat,
                          free_raw_data=False)
        dva = lgb.Dataset(_X(d[va], nat_codes), _y(d[va], target), weight=w[va.to_numpy()], categorical_feature=cat,
                          reference=dtr)
        b = lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        best = max(int(b.best_iteration or 0), 20)
    else:
        best = 200
    dall = lgb.Dataset(_X(d, nat_codes), _y(d, target), weight=w, categorical_feature=cat)
    booster = lgb.train(params, dall, num_boost_round=best)
    vm = [pd.Timestamp(m).strftime("%Y-%m") for m in months[-val_months:]] if cut is not None else []
    return GbmModel(booster=booster, target=target, nat_codes=nat_codes, best_iter=best, val_months=vm)


def raw_predict(m: GbmModel, df: pd.DataFrame) -> np.ndarray:
    return m.booster.predict(_X(df, m.nat_codes), num_threads=1)


def predict(m: GbmModel, df: pd.DataFrame) -> np.ndarray:
    f = raw_predict(m, df)
    if m.target == "ml_only":
        return np.expm1(f).clip(min=0)
    return df["l1_guests"].to_numpy(dtype=float) * np.exp(f)


def shap_values(m: GbmModel, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """TreeSHAP values (rows x features) on the model's own scale, via the shap package."""
    import shap

    X = _X(df, m.nat_codes)
    explainer = shap.TreeExplainer(m.booster)
    return np.asarray(explainer.shap_values(X)), X
