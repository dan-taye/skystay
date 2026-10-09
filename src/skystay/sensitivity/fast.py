"""Vectorised point evaluation of global lever settings (used by tornado and Sobol).

Same chain as the scenario engine (seats -> LF -> transfer -> conversion with
calendar uplift + non-air baseline -> ML correction -> ALOS), evaluated for
many lever settings at once. `tests/test_sensitivity.py` checks it reproduces
the engine's baseline and a frequency scenario exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skystay import settings
from skystay.chain import parametric as par
from skystay.scenario.engine import Engine

WINTER = "Winter peak (Dec-Feb)"
SUMMER = "Summer low (Jun-Aug)"


@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    lo: float
    hi: float
    neutral: float
    unit: str


FACTORS = [
    Factor("capacity", "Seat capacity (frequency x aircraft size)", 0.85, 1.15, 1.0, "x seats"),
    Factor("load_factor", "Load factor", -5.0, 5.0, 0.0, "pp"),
    Factor("transfer_share", "Transfer / transit share", -5.0, 5.0, 0.0, "pp"),
    Factor("capture", "Hotel capture rate", 0.9, 1.1, 1.0, "x"),
    Factor("alos", "Length of stay", 0.9, 1.1, 1.0, "x"),
    Factor("events", "Event days added per month", 0.0, 3.0, 0.0, "days"),
    Factor("season_shift", "Seasonal shift (summer seats moved to winter)", -15.0, 15.0, 0.0, "% of summer seats"),
    Factor("market_mix", "Market mix (seats moved from 5 lowest- to 5 highest-yield markets)", 0.0, 20.0, 0.0,
           "% of seats"),
]
KEYS = [f.key for f in FACTORS]


class FastEval:
    def __init__(self, engine: Engine):
        self.e = engine
        a = engine.baseline.arrays
        self.origins = list(a["origins"])
        self.S0 = a["S"]
        self.lf0, self.tr0 = a["lf"], a["tr"]
        m = engine.m
        self.conv = par.supply_arrays(m, self.origins)["conv"][:, :, engine.moy - 1]
        self.W = a["W"]
        self.base_raw = m.base[:, engine.moy - 1]
        self.alos = a["alos"]
        self.summer = engine.season_of == SUMMER
        self.winter = engine.season_of == WINTER
        # yield ranking (guests per seat) for the market-mix factor
        flow = a["flow"].sum(axis=(1, 2))
        seats = self.S0.sum(axis=1)
        gps = np.divide(flow, seats, out=np.zeros_like(flow), where=seats > 0)
        served = np.where(seats > 0)[0]
        order = served[np.argsort(gps[served])]
        self.low5, self.high5 = order[:5], order[-5:]

    def evaluate(self, X: np.ndarray, chunk: int = 1500) -> dict[str, np.ndarray]:
        """X: (K, 8) factor values in FACTORS order. Returns guests/nights (K, N, 12)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        outs_g, outs_n = [], []
        for s in range(0, len(X), chunk):
            g, n = self._eval(X[s:s + chunk])
            outs_g.append(g)
            outs_n.append(n)
        return {"guests": np.concatenate(outs_g), "nights": np.concatenate(outs_n)}

    def _eval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e, m = self.e, self.e.m
        K = len(X)
        cap_s, dlf, dtr, capm, alosm, ev_add, ss, mix = (X[:, i] for i in range(8))
        S = self.S0[None] * cap_s[:, None, None]
        # seasonal shift: positive moves summer seats into winter, negative the reverse
        for src, dst, sign in ((self.summer, self.winter, 1.0), (self.winter, self.summer, -1.0)):
            frac = np.clip(sign * ss, 0, None)[:, None] / 100.0
            if not frac.any():
                continue
            moved = (S[:, :, src] * frac[:, :, None]).sum(axis=2)  # (K, O)
            S[:, :, src] *= 1 - frac[:, :, None]
            wsh = S[:, :, dst] / np.maximum(S[:, :, dst].sum(axis=2, keepdims=True), 1e-9)
            S[:, :, dst] += moved[:, :, None] * wsh
        # market mix: move mix% of the 5 lowest-yield markets' seats to the 5 highest, month by month
        if mix.any():
            f = mix[:, None, None] / 100.0
            moved = (S[:, self.low5, :] * f).sum(axis=1)  # (K, 12)
            S[:, self.low5, :] *= 1 - f
            share = S[:, self.high5, :] / np.maximum(S[:, self.high5, :].sum(axis=1, keepdims=True), 1e-9)
            S[:, self.high5, :] += moved[:, None, :] * share
        lf = np.clip(self.lf0[None] + dlf[:, None, None] / 100.0, 0.05, 1.0)
        tr = np.clip(self.tr0[None] + dtr[:, None, None] / 100.0, 0.0, 0.95)
        T = S * lf * (1 - tr)
        ev = np.clip(e.event_days[None, :] + ev_add[:, None], 0, e.dim[None, :])  # (K, 12)
        ef = ev / e.dim[None, :]
        cm = 1 + m.hol_uplift * e.hf[None] + m.event_uplift_air * ef[:, None, :]  # (K, N, 12)
        cb = 1 + m.event_uplift_base * ef  # (K, 12)
        air = np.einsum("kom,onm->knm", T, self.conv, optimize=True) * cm * capm[:, None, None]
        base = self.base_raw[None] * cb[:, None, :]
        l1 = air + base
        vis = np.einsum("kom,onm->knm", T, self.W, optimize=True)
        factor = self._factor(vis, ev)
        guests = l1 * factor if e.model_name != "ml_only" else factor
        nights = guests * self.alos[None] * alosm[:, None, None]
        return guests, nights

    def _factor(self, vis: np.ndarray, ev: np.ndarray) -> np.ndarray:
        e = self.e
        K = vis.shape[0]
        if e.gbm is None:
            return np.ones_like(vis)
        X = np.column_stack([
            np.tile(e._X["nat_code"], K), np.tile(e._X["moy"], K), np.tile(e._X["holiday_days"], K),
            np.repeat(ev, e.N, axis=0).ravel(), np.tile(e._X["trend"], K),
            np.log1p(np.clip(vis, 0, None)).reshape(-1), np.tile(e._X["lag12"], K),
        ])
        f = e.gbm.booster.predict(X, num_threads=1).reshape(K, e.N, 12)
        return np.exp(f) if e.model_name != "ml_only" else np.expm1(f).clip(min=0)

    def neutral(self) -> np.ndarray:
        return np.array([f.neutral for f in FACTORS])


def historical_sd(engine: Engine) -> dict[str, float]:
    """One historical standard deviation per lever, from year-to-year variation in the data."""
    p = engine.p
    om = p.origin_month.assign(year=p.origin_month["month"].dt.year)
    nm = p.nat_month.assign(year=p.nat_month["month"].dt.year)
    y = om.groupby("year").agg(seats=("seats", "sum"), pax=("pax", "sum"), transfer=("transfer", "sum"))
    y = y[y.index.isin(sorted(y.index)[:])]
    seats_growth = y["seats"].pct_change().dropna()
    lf = y["pax"] / y["seats"]
    trs = y["transfer"] / y["pax"]
    ny = nm.groupby("year").agg(g=("guests", "sum"), n=("nights", "sum"))
    alos = ny["n"] / ny["g"]
    conv = (ny["g"] / (y["pax"] - y["transfer"])).dropna()
    ev = p.nat_month.drop_duplicates("month").assign(year=lambda d: d["month"].dt.year).groupby("year")[
        "event_days"].sum() / 12
    return {
        "capacity": float(seats_growth.std()),  # relative
        "load_factor": float(lf.std() * 100),  # pp
        "transfer_share": float(trs.std() * 100),  # pp
        "capture": float((conv.pct_change().dropna()).std()),  # relative
        "alos": float(alos.pct_change().dropna().std()),  # relative
        "events": float(ev.std()),  # days per month
    }


def season_names() -> list[str]:
    return list(settings.seasons().keys())
