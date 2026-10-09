"""Layer 2: econometric baseline (statsmodels OLS, HC3 standard errors).

    log(G_n,t) = b * log(1 + air visitors_n,t) + month FE + market FE
                 + c_h * holiday share + c_e * event share + e

Both a benchmark and a sanity check on scenario responses: b is the elasticity
of hotel guests to air visitors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

FORMULA = "np.log(guests) ~ log_air + C(moy) + C(nationality) + hol_frac + evt_frac"


@dataclass
class EconModel:
    result: object
    smear: float
    nationalities: list[str]

    def summary(self) -> dict:
        r = self.result
        ci = r.conf_int(alpha=0.05)
        out = {}
        for name, key in (("elasticity", "log_air"), ("holiday_share", "hol_frac"), ("event_share", "evt_frac")):
            out[name] = {"coef": float(r.params[key]), "ci95": [float(ci.loc[key, 0]), float(ci.loc[key, 1])],
                         "se_hc3": float(r.bse[key])}
        out["n_obs"] = int(r.nobs)
        out["r2"] = float(r.rsquared)
        out["cov_type"] = "HC3"
        return out


def fit(train: pd.DataFrame) -> EconModel:
    d = train[(train["guests"] > 0) & train["log_air"].notna()].copy()
    d["nationality"] = d["nationality"].astype(str)
    res = smf.ols(FORMULA, data=d).fit(cov_type="HC3")
    smear = float(np.mean(np.exp(res.resid)))  # Duan smearing for the retransformation bias
    return EconModel(result=res, smear=smear, nationalities=sorted(d["nationality"].unique()))


def predict(m: EconModel, df: pd.DataFrame) -> np.ndarray:
    d = df.copy()
    d["nationality"] = d["nationality"].astype(str)
    known = d["nationality"].isin(m.nationalities)
    out = np.full(len(d), np.nan)
    if known.any():
        out[known.to_numpy()] = np.exp(m.result.predict(d[known])) * m.smear
    return out
