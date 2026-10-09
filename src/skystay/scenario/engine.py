"""What-if scenario engine.

A scenario = the reference year (the latest 12 months of schedule, traffic
parameters and calendars) + a list of lever changes. `run` recomputes seats,
passengers, visitors, hotel guests and guest nights by market and month, with
Monte Carlo uncertainty and deltas against the baseline.

Uncertainty (500 draws by default, common random numbers so deltas are
consistent): load factor and transfer share from their EB standard errors,
capture level and non-air baseline from the year-block bootstrap, length of
stay from its EB standard error, plus the split-conformal model error (log
scale, half common to the year and half month-specific). Cold-start origins
get doubled parameter uncertainty; markets pushed outside their historical
range get their model error widened by 1.5x.

Levers move only the air-attributable part of demand; the non-air baseline
responds only to the event lever.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from skystay import settings
from skystay.chain import parametric as par
from skystay.ml import features, gbm
from skystay.reconcile import regions
from skystay.scenario.levers import (
    Aircraft,
    Alos,
    Capture,
    DiscontinueRoute,
    Event,
    Frequency,
    LoadFactor,
    MixShift,
    NewRoute,
    Scenario,
    SeasonShift,
    TransferShare,
)

Z80, Z95 = 1.2816, 1.96
EXTRAP_SEATS = 1.15  # monthly seats above 115% of the historical max (or below 85% of the min) = extrapolation
EXTRAP_ERROR_INFLATION = 1.5
COLD_PARAM_INFLATION = 2.0
COLD_CONV_SD = 0.25  # extra log-SD on a cold-start origin's conversion (region-borrowed)


class ScenarioError(ValueError):
    """An impossible input (e.g. a load factor above 100%)."""


@dataclass
class ScenarioResult:
    scenario: Scenario
    table: pd.DataFrame
    monthly: pd.DataFrame
    totals: dict
    warnings: list[str]
    routes: pd.DataFrame
    arrays: dict = field(repr=False, default_factory=dict)


class Engine:
    def __init__(self, bundle, draws: int | None = None):
        self.b = bundle
        self.m: par.ParametricModel = bundle.chain
        self.p = bundle.panels
        self.model_name = bundle.model_name if bundle.model_name in ("parametric", "hybrid", "ml_only") else "parametric"
        self.gbm = bundle.gbm if self.model_name in ("hybrid", "ml_only") else None
        self.months = self.p.months[-settings.load_settings()["scenario"]["reference_months"]:]
        self.moy = self.months.month.to_numpy()
        self.dim = self.months.days_in_month.to_numpy().astype(float)
        self.nats = self.m.nationalities
        self.N = len(self.nats)
        self.season_of = np.array([settings.season_of_month(int(k)) for k in self.moy])
        self.draws = draws or settings.load_settings()["scenario"]["draws"]
        self._routes()
        self._calendar()
        self._features()
        self._conformal()
        self._history()
        self.baseline = self.run(Scenario(name="Baseline", draws=self.draws), _is_baseline=True)

    # ------------------------------------------------------------------ setup
    def _routes(self) -> None:
        r = self.p.routes[self.p.routes["month"].isin(self.months)].copy()
        r["route_id"] = r["origin_airport"] + " · " + r["airline"]
        g = r.groupby(["route_id", "month"]).agg(
            origin=("origin_country", "first"), airport=("origin_airport", "first"), airline=("airline", "first"),
            aircraft=("aircraft_type", "last"), flights=("flights", "sum"), seats=("seats", "sum"),
        ).reset_index()
        g["new"] = False
        g[["flights", "seats"]] = g[["flights", "seats"]].astype(float)
        self.routes = g
        allr = self.p.routes
        spf = allr.groupby("aircraft_type").apply(lambda x: x["seats"].sum() / max(x["flights"].sum(), 1),
                                                  include_groups=False)
        self.spf = {k: float(v) for k, v in spf.items()}

    def _calendar(self) -> None:
        cal = par.calendar_frame(self.p, self.months)
        hf = (cal["holiday_days"] / cal["dim"]).to_numpy()
        self.hf = (cal.assign(hf=hf).pivot_table(index="nationality", columns="month", values="hf")
                   .reindex(index=self.nats, columns=self.months).fillna(0.0).to_numpy())
        self.hol_days = (cal.pivot_table(index="nationality", columns="month", values="holiday_days")
                         .reindex(index=self.nats, columns=self.months).fillna(0.0).to_numpy())
        self.event_days = cal.groupby("month")["event_days"].max().reindex(self.months).fillna(0.0).to_numpy()

    def _features(self) -> None:
        if self.gbm is None:
            return
        f = features.frame(self.p, self.m, self.months, use_actual_supply=False)
        f = f.set_index(["nationality", "month"]).reindex(
            pd.MultiIndex.from_product([self.nats, self.months])).reset_index()
        codes = np.array([self.gbm.nat_codes.get(n, -1) for n in self.nats], dtype=float)
        self._X = {
            "nat_code": np.repeat(codes, 12),
            "moy": np.tile(self.moy, self.N).astype(float),
            "holiday_days": self.hol_days.ravel(),
            "trend": f["trend"].to_numpy(dtype=float),
            "lag12": f["lag12"].to_numpy(dtype=float),
        }

    def _conformal(self) -> None:
        q = (self.b.conformal or {}).get("guests", {})
        allq = q.get("_all", {"0.8": [-0.1, 0.1]})
        mu, sd = np.zeros(self.N), np.zeros(self.N)
        for j, n in enumerate(self.nats):
            lo, hi = q.get(n, allq)["0.8"]
            mu[j] = (lo + hi) / 2
            sd[j] = max((hi - lo) / (2 * Z80), 1e-4)
        self.err_mu, self.err_sd = mu, sd

    def _history(self) -> None:
        om = self.p.origin_month
        served = om[om["seats"] > 0]
        self.hist_seats = served.groupby("origin")["seats"].agg(["min", "max"]).to_dict("index")
        self.hist_lf = served.groupby("origin")["lf"].agg(["min", "max"]).to_dict("index")
        m = self.m
        self.cap_se = np.zeros(self.N)
        self.base_se = np.zeros(self.N)
        reg = regions.region_of()
        for j, n in enumerate(self.nats):
            u = n if n in m.units else reg[n]
            if u in m.units:
                k = m.units.index(u)
                if m.unit_cap_se_rel is not None:
                    self.cap_se[j] = m.unit_cap_se_rel[k]
                if m.unit_base_se_rel is not None:
                    self.base_se[j] = m.unit_base_se_rel[k]
        self.cap_se = np.clip(np.nan_to_num(self.cap_se), 0.0, 1.0)
        self.base_se = np.clip(np.nan_to_num(self.base_se), 0.0, 1.0)
        self.alos_se_rel = np.clip((m.alos_se / np.maximum(m.alos, 1e-6)).mean(axis=1), 0, 0.5)

    # ------------------------------------------------------------------ helpers
    def season_mask(self, season: str) -> np.ndarray:
        if season == "ALL":
            return np.ones(12, dtype=bool)
        if season not in settings.seasons():
            raise ScenarioError(f"unknown season '{season}'")
        return self.season_of == season

    def _market_mask(self, origins: list[str], market: str) -> np.ndarray:
        if market == "ALL":
            return np.ones(len(origins), dtype=bool)
        return np.array([o == market for o in origins])

    def _nat_mask(self, market: str) -> np.ndarray:
        if market == "ALL":
            return np.ones(self.N, dtype=bool)
        if market not in self.nats:
            raise ScenarioError(f"unknown nationality '{market}'")
        return np.array([n == market for n in self.nats])

    def seats_per_flight(self, aircraft: str) -> float:
        if aircraft not in self.spf:
            raise ScenarioError(f"unknown aircraft type '{aircraft}' (known: {sorted(self.spf)})")
        return self.spf[aircraft]

    # ------------------------------------------------------------------ levers -> seats
    def _apply_route_levers(self, sc: Scenario) -> pd.DataFrame:
        r = self.routes.copy()
        r["moy"] = r["month"].dt.month
        for lv in sc.levers:
            if isinstance(lv, NewRoute):
                spf = lv.seats_per_flight or self.seats_per_flight(lv.aircraft)
                rows = []
                for k, mth in enumerate(self.months):
                    if mth.month < lv.start_month:
                        continue
                    flights = lv.weekly_freq * self.dim[k] / 7.0
                    rows.append({"route_id": f"NEW {lv.origin} · {lv.airline}", "month": mth, "origin": lv.origin,
                                 "airport": f"NEW-{lv.origin}", "airline": lv.airline, "aircraft": lv.aircraft,
                                 "flights": flights, "seats": flights * spf, "new": True, "moy": mth.month})
                if rows:
                    r = pd.concat([r, pd.DataFrame(rows)], ignore_index=True)
            elif isinstance(lv, DiscontinueRoute):
                hit = r["route_id"] == lv.route_id
                if not hit.any():
                    raise ScenarioError(f"unknown route '{lv.route_id}'")
                cut = hit & (r["moy"] >= lv.from_month)
                r.loc[cut, ["flights", "seats"]] = 0.0
            elif isinstance(lv, Frequency):
                sm = self.season_mask(lv.season)
                hit = ((r["origin"] == lv.market) | (lv.market == "ALL")) & r["moy"].isin(self.moy[sm])
                r.loc[hit, ["flights", "seats"]] *= 1 + lv.pct / 100.0
            elif isinstance(lv, Aircraft):
                hit = r["origin"] == lv.market
                if lv.aircraft:
                    spf = self.seats_per_flight(lv.aircraft)
                    r.loc[hit, "seats"] = r.loc[hit, "flights"] * spf
                    r.loc[hit, "aircraft"] = lv.aircraft
                else:
                    r.loc[hit, "seats"] *= 1 + lv.seats_pct / 100.0
            elif isinstance(lv, MixShift):
                sm = self.season_mask(lv.season)
                for mth in self.months[sm]:
                    src = (r["origin"] == lv.from_market) & (r["month"] == mth)
                    moved = r.loc[src, "seats"].sum() * lv.pct / 100.0
                    if moved <= 0:
                        continue
                    r.loc[src, ["seats", "flights"]] *= 1 - lv.pct / 100.0
                    dst = (r["origin"] == lv.to_market) & (r["month"] == mth)
                    if r.loc[dst, "seats"].sum() > 0:
                        r.loc[dst, "seats"] += moved * r.loc[dst, "seats"] / r.loc[dst, "seats"].sum()
                    else:
                        r = pd.concat([r, pd.DataFrame([{
                            "route_id": f"NEW {lv.to_market} · reallocated", "month": mth, "origin": lv.to_market,
                            "airport": f"NEW-{lv.to_market}", "airline": "reallocated", "aircraft": "mixed",
                            "flights": 0.0, "seats": moved, "new": True, "moy": mth.month}])], ignore_index=True)
            elif isinstance(lv, SeasonShift):
                fm, tm = self.season_mask(lv.from_season), self.season_mask(lv.to_season)
                sel = (r["origin"] == lv.market) | (lv.market == "ALL")
                src = sel & r["moy"].isin(self.moy[fm])
                moved = r.loc[src, "seats"].sum() * lv.pct / 100.0
                r.loc[src, ["seats", "flights"]] *= 1 - lv.pct / 100.0
                dst = sel & r["moy"].isin(self.moy[tm])
                tot = r.loc[dst, "seats"].sum()
                if tot > 0 and moved > 0:
                    share = r.loc[dst, "seats"] / tot
                    r.loc[dst, "flights"] += moved * share / np.maximum(r.loc[dst, "seats"] / np.maximum(
                        r.loc[dst, "flights"], 1e-9), 1.0)
                    r.loc[dst, "seats"] += moved * share
        if (r["seats"] < -1e-9).any() or (r["flights"] < -1e-9).any():
            raise ScenarioError("negative frequency or seats after applying the levers")
        return r

    # ------------------------------------------------------------------ core
    def run(self, sc: Scenario, _is_baseline: bool = False) -> ScenarioResult:
        warnings: list[str] = []
        r = self._apply_route_levers(sc)
        seats = r.groupby(["origin", "month"])["seats"].sum().unstack("month").reindex(columns=self.months).fillna(0.0)
        base_origins = list(self.baseline.arrays["origins"]) if hasattr(self, "baseline") else []
        origins = sorted(set(seats.index) | set(base_origins))
        S = seats.reindex(origins).fillna(0.0).to_numpy()
        arr = par.supply_arrays(self.m, origins)
        lf, tr = arr["lf"][:, self.moy - 1].copy(), arr["tr"][:, self.moy - 1].copy()
        lf_se = arr["lf_se"].mean(axis=1)
        tr_se = arr["tr_se"].mean(axis=1)
        cold = arr["cold"] & (S.sum(axis=1) > 0)
        conv = arr["conv"][:, :, self.moy - 1].copy()  # (O, N, 12)

        cap_mult = np.ones(self.N)
        alos_mult = np.ones(self.N)
        ev = self.event_days.copy()
        for lv in sc.levers:
            if isinstance(lv, LoadFactor):
                mk = self._market_mask(origins, lv.market)
                lf[mk] = lv.value if lv.value is not None else lf[mk] + lv.delta_pp / 100.0
            elif isinstance(lv, TransferShare):
                mk = self._market_mask(origins, lv.market)
                tr[mk] = tr[mk] + lv.delta_pp / 100.0
            elif isinstance(lv, Capture):
                cap_mult[self._nat_mask(lv.market)] *= 1 + lv.pct / 100.0
                if abs(lv.pct) > 20:
                    warnings.append(f"extrapolation: capture change of {lv.pct:+.0f}% for {lv.market} is beyond the "
                                    "variation seen in the data")
            elif isinstance(lv, Alos):
                alos_mult[self._nat_mask(lv.market)] *= 1 + lv.pct / 100.0
                if abs(lv.pct) > 20:
                    warnings.append(f"extrapolation: length-of-stay change of {lv.pct:+.0f}% for {lv.market} is "
                                    "beyond the variation seen in the data")
            elif isinstance(lv, Event):
                k = lv.month - 1
                ev[k] = float(np.clip(ev[k] + lv.days, 0, self.dim[k]))
        active = S > 0
        if (lf[active] > 1.0 + 1e-9).any():
            bad = sorted({origins[i] for i in np.where((lf > 1.0 + 1e-9) & active)[0]})
            raise ScenarioError(f"impossible input: load factor above 100% for {', '.join(bad)}")
        if (lf[active] < 0.05).any():
            raise ScenarioError("impossible input: load factor below 5%")
        if (tr[active] < 0).any() or (tr[active] > 0.95).any():
            raise ScenarioError("impossible input: transfer share outside 0-95%")
        tr = np.clip(tr, 0.0, 0.95)

        # extrapolation and cold-start flags
        extrap_o = np.zeros(len(origins), dtype=bool)
        for i, o in enumerate(origins):
            if cold[i]:
                warnings.append(f"cold-start estimate for {regions.label(o)} ({o}): no service history, so load "
                                "factor, transfer share and conversion are borrowed from its region; uncertainty "
                                "doubled")
                continue
            h = self.hist_seats.get(o)
            if h and active[i].any():
                mx = S[i][active[i]].max()
                mn = S[i][active[i]].min()
                if mx > h["max"] * EXTRAP_SEATS or mn < h["min"] * (2 - EXTRAP_SEATS):
                    extrap_o[i] = True
                    warnings.append(f"extrapolation: monthly seats for {regions.label(o)} leave the historical range "
                                    f"({h['min']:,.0f} to {h['max']:,.0f})")
            hl = self.hist_lf.get(o)
            if hl and active[i].any():
                lv_ = lf[i][active[i]]
                if lv_.max() > hl["max"] + 0.02 or lv_.min() < hl["min"] - 0.02:
                    extrap_o[i] = True
                    warnings.append(f"extrapolation: load factor for {regions.label(o)} outside its historical range "
                                    f"({hl['min']:.0%} to {hl['max']:.0%})")

        m = self.m
        hf = self.hf
        ef = ev / self.dim
        cm = 1 + m.hol_uplift * hf + m.event_uplift_air * ef[None, :]
        cb = 1 + m.event_uplift_base * ef
        base = m.base[:, self.moy - 1] * cb[None, :]
        alos = m.alos[:, self.moy - 1] * alos_mult[:, None]
        convc = conv * cap_mult[None, :, None]
        tot = conv.sum(axis=1, keepdims=True)
        W = np.divide(conv, tot, out=np.zeros_like(conv), where=tot > 0)  # nationality mix per origin

        P = S * lf
        T = P * (1 - tr)
        flow = T[:, None, :] * convc * cm[None, :, :]  # (O, N, 12) air guests by origin x nationality
        air = flow.sum(axis=0)
        visitors = np.einsum("om,onm->nm", T, W)
        l1 = air + base
        factor = self._ml_factor(visitors, ev, l1)
        guests = l1 * factor if self.model_name != "ml_only" else factor
        nights = guests * alos
        seats_n = np.einsum("om,onm->nm", S, W)
        pax_n = np.einsum("om,onm->nm", P, W)

        linked_cold = (W[cold].sum(axis=0) > 0).any(axis=1) if cold.any() else np.zeros(self.N, dtype=bool)
        linked_ext = (W[extrap_o].sum(axis=0) > 0).any(axis=1) if extrap_o.any() else np.zeros(self.N, dtype=bool)

        # ---------------- Monte Carlo ----------------
        D = sc.draws
        z = self._draws(sc.seed, D, origins)
        lf_d = np.clip(lf[None] + (z["lf"] * (lf_se * np.where(cold, COLD_PARAM_INFLATION, 1.0)))[:, :, None],
                       0.05, 1.0)
        tr_d = np.clip(tr[None] + (z["tr"] * (tr_se * np.where(cold, COLD_PARAM_INFLATION, 1.0)))[:, :, None],
                       0.0, 0.95)
        T_d = S[None] * lf_d * (1 - tr_d) * np.exp(z["cold"] * COLD_CONV_SD * cold[None, :])[:, :, None]
        air_d = np.einsum("dom,onm->dnm", T_d, convc * cm[None], optimize=True)
        air_d *= np.exp(z["cap"] * self.cap_se[None, :])[:, :, None]
        base_d = base[None] * np.exp(z["base"] * self.base_se[None, :])[:, :, None]
        infl = np.where(linked_ext | linked_cold, EXTRAP_ERROR_INFLATION, 1.0)
        err = self.err_mu[None, :, None] + (self.err_sd * infl)[None, :, None] * (
            np.sqrt(0.5) * z["err_c"][:, :, None] + np.sqrt(0.5) * z["err_m"])
        if self.model_name == "ml_only":
            g_d = guests[None] * (air_d + base_d) / np.maximum(l1[None], 1e-9) * np.exp(err)
        else:
            g_d = (air_d + base_d) * factor[None] * np.exp(err)
        n_d = g_d * alos[None] * (1 + z["alos"] * self.alos_se_rel[None, :])[:, :, None]

        q = np.percentile(g_d, [10, 90, 2.5, 97.5], axis=0)
        qn = np.percentile(n_d, [10, 90, 2.5, 97.5], axis=0)
        idx = pd.MultiIndex.from_product([self.months, self.nats], names=["month", "market"])
        tab = pd.DataFrame(index=idx).reset_index()

        def flat(a):  # (N, 12) -> month-major rows
            return a.T.ravel()

        tab["market_name"] = tab["market"].map(regions.label)
        tab["seats"] = flat(seats_n)
        tab["pax"] = flat(pax_n)
        tab["visitors"] = flat(visitors)
        tab["air_guests"] = flat(air * factor if self.model_name != "ml_only" else guests - base)
        tab["guests"] = flat(guests)
        tab["nights"] = flat(nights)
        tab["lo80"], tab["hi80"], tab["lo95"], tab["hi95"] = (flat(q[k]) for k in range(4))
        tab["nights_lo80"], tab["nights_hi80"], tab["nights_lo95"], tab["nights_hi95"] = (flat(qn[k]) for k in range(4))
        tab["cold_start"] = np.tile(linked_cold, 12)
        tab["extrapolation"] = np.tile(linked_ext, 12)

        bl = None if (_is_baseline or not hasattr(self, "baseline")) else self._baseline_for(sc)
        if bl is not None:
            bg, bn = bl.arrays["guests"], bl.arrays["nights"]
            bgd, bnd = bl.arrays["g_d"], bl.arrays["n_d"]
        else:
            bg, bn, bgd, bnd = guests, nights, g_d, n_d
        tab["delta_vs_baseline"] = flat(guests - bg)
        tab["delta_nights"] = flat(nights - bn)

        mt = pd.DataFrame({"month": self.months, "guests": guests.sum(axis=0), "nights": nights.sum(axis=0),
                           "baseline_guests": bg.sum(axis=0), "baseline_nights": bn.sum(axis=0)})
        gm = g_d.sum(axis=1)
        nm = n_d.sum(axis=1)
        for name, a in (("guests", gm), ("nights", nm)):
            for lab, pr in (("lo80", 10), ("hi80", 90), ("lo95", 2.5), ("hi95", 97.5)):
                mt[f"{name}_{lab}"] = np.percentile(a, pr, axis=0)
        ann_g, ann_n = gm.sum(axis=1), nm.sum(axis=1)
        dg = ann_g - bgd.sum(axis=(1, 2))
        dn = ann_n - bnd.sum(axis=(1, 2))

        def iv(a):
            return {"lo80": float(np.percentile(a, 10)), "hi80": float(np.percentile(a, 90)),
                    "lo95": float(np.percentile(a, 2.5)), "hi95": float(np.percentile(a, 97.5))}

        totals = {
            "guests": float(guests.sum()), "nights": float(nights.sum()),
            "guests_interval": iv(ann_g), "nights_interval": iv(ann_n),
            "delta_guests": float(guests.sum() - bg.sum()), "delta_nights": float(nights.sum() - bn.sum()),
            "delta_guests_interval": iv(dg), "delta_nights_interval": iv(dn),
            "seats": float(S.sum()), "baseline_seats": float(bl.totals["seats"]) if bl else float(S.sum()),
            "air_guests": float(tab["air_guests"].sum()), "non_air_guests": float(base.sum()),
        }
        arrays = {"origins": origins, "S": S, "P": P, "T": T, "flow": flow, "factor": factor, "base": base,
                  "guests": guests, "nights": nights, "g_d": g_d, "n_d": n_d, "lf": lf, "tr": tr, "cold": cold,
                  "alos": alos, "W": W}
        return ScenarioResult(scenario=sc, table=tab, monthly=mt, totals=totals, warnings=list(dict.fromkeys(warnings)),
                              routes=r, arrays=arrays)

    def _baseline_for(self, sc: Scenario) -> ScenarioResult:
        """Baseline with the same draws and seed (common random numbers for the deltas)."""
        if sc.draws == self.baseline.scenario.draws and sc.seed == self.baseline.scenario.seed:
            return self.baseline
        cache = self.__dict__.setdefault("_bl_cache", {})
        key = (sc.draws, sc.seed)
        if key not in cache:
            cache[key] = self.run(Scenario(name="Baseline", draws=sc.draws, seed=sc.seed), _is_baseline=True)
        return cache[key]

    def _ml_factor(self, visitors: np.ndarray, event_days: np.ndarray, l1: np.ndarray) -> np.ndarray:
        if self.gbm is None:
            return np.ones_like(l1)
        X = np.column_stack([
            self._X["nat_code"], self._X["moy"], self._X["holiday_days"], np.tile(event_days, self.N),
            self._X["trend"], np.log1p(np.clip(visitors, 0, None)).ravel(), self._X["lag12"],
        ])
        f = self.gbm.booster.predict(X, num_threads=1).reshape(self.N, 12)
        if self.model_name == "ml_only":
            return np.expm1(f).clip(min=0)
        return np.exp(f)

    def _draws(self, seed: int, D: int, origins: list[str]) -> dict[str, np.ndarray]:
        """Standard-normal draws keyed by entity, so adding a route never reshuffles other markets' draws."""
        rng = np.random.default_rng(seed)
        out = {
            "cap": rng.standard_normal((D, self.N)),
            "base": rng.standard_normal((D, self.N)),
            "alos": rng.standard_normal((D, self.N)),
            "err_c": rng.standard_normal((D, self.N)),
            "err_m": rng.standard_normal((D, self.N, 12)),
        }
        cols = {"lf": [], "tr": [], "cold": []}
        for o in origins:
            r = np.random.default_rng([seed, zlib.crc32(o.encode())])
            cols["lf"].append(r.standard_normal(D))
            cols["tr"].append(r.standard_normal(D))
            cols["cold"].append(r.standard_normal(D))
        for k, v in cols.items():
            out[k] = np.column_stack(v) if v else np.zeros((D, 0))
        return out

    # ------------------------------------------------------------------ helpers for the UI / sensitivity
    def route_table(self) -> pd.DataFrame:
        g = self.routes.groupby("route_id").agg(origin=("origin", "first"), airline=("airline", "first"),
                                                 aircraft=("aircraft", "last"), seats=("seats", "sum"),
                                                 flights=("flights", "sum"))
        g["weekly_freq"] = g["flights"] / (self.dim.sum() / 7.0)
        return g.reset_index().sort_values(["origin", "route_id"])

    def unserved_markets(self) -> list[str]:
        served = set(self.routes.loc[self.routes["seats"] > 0, "origin"])
        reg = regions.region_of()
        return sorted(k for k in reg if k not in served and k not in ("AE", "OTHER"))


def gbm_features() -> list[str]:
    return list(gbm.FEATURES)
