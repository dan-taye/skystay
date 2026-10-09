"""`skystay train`: fit the production model on all months and save the bundle.

Uses the (reconciliation level, model) selected by `skystay backtest`
(reports/metrics.json). Also always fits the L2 bridge (for the bridge heatmap
and the parameter-recovery test), the econometric layer (elasticity sanity
check) and SHAP values for the LightGBM layer. Writes:

- data/processed/model/bundle.pkl   (git-ignored; consumed by the app and scenario engine)
- config/assumptions.yaml           (generated assumption ledger)
- docs/ASSUMPTIONS.md               (human-readable ledger)
- reports/model.json, reports/shap.json, reports/recovery.json (synthetic mode)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from skystay import paths, settings
from skystay.chain import assumptions, panel
from skystay.chain import parametric as par
from skystay.ml import econometric, features, gbm
from skystay.reconcile import bridge, regions

BUNDLE = "bundle.pkl"


@dataclass
class Bundle:
    level: str
    model_name: str
    chain: par.ParametricModel
    l2: par.ParametricModel
    gbm: gbm.GbmModel | None
    econ: dict
    conformal: dict
    panels: panel.Panels
    selection: dict
    data_mode: str
    meta: dict = field(default_factory=dict)


def load_selection() -> dict:
    f = paths.REPORTS / "metrics.json"
    if f.exists():
        m = json.loads(f.read_text(encoding="utf-8"))
        return {"level": m["selection"]["level"], "model": m["selection"]["model"], "source": "reports/metrics.json",
                "conformal": m["chosen"]["conformal"]}
    return {"level": "L1", "model": "hybrid", "source": "default (no back-test yet)", "conformal": {}}


def run(bootstrap: int = 100) -> Bundle:
    paths.ensure_dirs()
    sel = load_selection()
    level, model_name = sel["level"], sel["model"]
    print(f"production model: {level} / {model_name} ({sel['source']})")
    p = panel.build()
    chain = par.fit(p, level, bootstrap=bootstrap)
    l2 = chain if level == "L2" else par.fit(p, "L2", bootstrap=bootstrap)

    feat_seats = features.frame(p, chain, p.months, use_actual_supply=False)
    feat_act = features.frame(p, chain, p.months, use_actual_supply=True)
    econ = econometric.fit(feat_act)
    econ_summary = econ.summary()

    gm = None
    shap_out = {}
    if model_name in ("ml_only", "hybrid"):
        gm = gbm.fit(feat_seats, model_name)
        sv, X = gbm.shap_values(gm, feat_seats[feat_seats["guests"] > 0])
        imp = np.abs(sv).mean(axis=0)
        rng = np.random.default_rng(42)
        pick = rng.choice(len(X), size=min(600, len(X)), replace=False)
        shap_out = {
            "target": gm.target,
            "best_iteration": gm.best_iter,
            "features": gbm.FEATURES,
            "mean_abs_shap": dict(zip(gbm.FEATURES, imp.round(5).tolist(), strict=True)),
            "sample": {"shap": sv[pick].round(4).tolist(), "x": X.iloc[pick].round(4).to_numpy().tolist()},
            "nat_codes": gm.nat_codes,
            "monotone": dict(zip(gbm.FEATURES, gbm.MONOTONE, strict=True)),
        }
        (paths.REPORTS / "shap.json").write_text(json.dumps(shap_out, indent=1), encoding="utf-8")

    b = Bundle(level=level, model_name=model_name, chain=chain, l2=l2, gbm=gm, econ=econ_summary,
               conformal=sel["conformal"], panels=p, selection=sel, data_mode=settings.data_mode(),
               meta={"fitted_on": f"{p.months[0]:%Y-%m}..{p.months[-1]:%Y-%m}", "bootstrap_reps": bootstrap})
    with open(paths.MODEL_DIR / BUNDLE, "wb") as fh:
        pickle.dump(b, fh)

    summary = model_summary(b)
    (paths.REPORTS / "model.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    assumptions.write(b, summary)
    if settings.is_synthetic():
        rec = recovery(b)
        (paths.REPORTS / "recovery.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
        print("parameter recovery:", {k: v["pass"] for k, v in rec["checks"].items()})
    print(f"saved {paths.MODEL_DIR.relative_to(paths.ROOT) / BUNDLE}")
    return b


def load() -> Bundle:
    f = paths.MODEL_DIR / BUNDLE
    if not f.exists():
        raise FileNotFoundError("model bundle missing: run `skystay train` (or `skystay all`)")
    with open(f, "rb") as fh:
        return pickle.load(fh)


def model_summary(b: Bundle) -> dict:
    """Headline parameters of the production model (all derived from the fit)."""
    p, m = b.panels, b.chain
    nas = par.non_air_share(m, p)
    last = p.months[-12:]
    pred = par.predict(m, par.seats_frame(p, last), par.terminating_frame(p, last), par.calendar_frame(p, last))
    g = pred.groupby("nationality")[["guests_pred", "air_guests", "base_guests", "air_visitors", "nights_pred"]].sum()
    total = g.sum()
    cap = (g["air_guests"] / g["air_visitors"].replace(0, np.nan)).to_dict()
    se_rel = {}
    if m.unit_cap_se_rel is not None:
        units = m.units
        reg = regions.region_of()
        for n in m.nationalities:
            u = n if n in units else reg[n]
            if u in units:
                se_rel[n] = float(m.unit_cap_se_rel[units.index(u)])
    l2b = b.l2.bridge or {}
    K = l2b.get("K")
    eff = K * l2b["m_hat"].mean(axis=1)[None, :] if K is not None else None
    mix = bridge.mix_matrix(eff) if eff is not None else None
    bridge_out = {}
    if mix is not None:
        bs = l2b.get("bootstrap", {})
        bridge_out = {
            "origins": b.l2.origins,
            "nationalities": b.l2.nationalities,
            "mix": mix.round(4).tolist(),
            "mix_lo80": bs["mix_lo"].round(4).tolist() if "mix_lo" in bs else None,
            "mix_hi80": bs["mix_hi"].round(4).tolist() if "mix_hi" in bs else None,
            "guests_per_terminating_pax": eff.sum(axis=1).round(4).tolist(),
            "lambda": l2b.get("lambda"),
            "lambda_scores": {str(k): v for k, v in (l2b.get("lambda_scores") or {}).items()},
            "bootstrap_reps": bs.get("reps"),
            "diagonal_mix": {o: float(mix[i, b.l2.nationalities.index(o)]) for i, o in enumerate(b.l2.origins)
                             if o in b.l2.nationalities},
            "identification_note": "CAP, B and returning-resident shares are not separately identified from "
                                   "aggregated data; we report K (guests per terminating passenger) and its "
                                   "row-normalised mix.",
        }
    return {
        "data_mode": b.data_mode,
        "level": b.level,
        "model": b.model_name,
        "fitted_on": b.meta["fitted_on"],
        "last12": {
            "guests": float(total["guests_pred"]), "nights": float(total["nights_pred"]),
            "air_guests": float(total["air_guests"]), "non_air_guests": float(total["base_guests"]),
            "non_air_share_total": float(total["base_guests"] / total["guests_pred"]),
        },
        "non_air_share": nas,
        "capture_per_visitor": {k: (None if pd.isna(v) else float(v)) for k, v in cap.items()},
        "capture_se_rel": se_rel,
        "calendar": {"holiday_uplift": m.hol_uplift, "event_uplift_air": m.event_uplift_air,
                     "event_uplift_base": m.event_uplift_base},
        "econometric": b.econ,
        "bridge": bridge_out,
        "origins": m.origins,
        "nationalities": m.nationalities,
        "notes": m.notes,
    }


def recovery(b: Bundle) -> dict:
    """Compare recovered parameters with the synthetic ground truth (stated tolerances)."""
    truth = json.loads((paths.SYNTH_GEN / "ground_truth.json").read_text(encoding="utf-8"))
    m, l2, p = b.chain, b.l2, b.panels
    checks = {}

    def add(name, errors: dict, stat: str, tol: float, desc: str):
        vals = np.array(list(errors.values()), dtype=float)
        value = float(np.max(np.abs(vals))) if stat == "max_abs" else float(np.median(np.abs(vals)))
        checks[name] = {"statistic": stat, "value": value, "tolerance": tol, "pass": bool(value <= tol),
                        "description": desc, "errors": {k: float(v) for k, v in errors.items()}}

    add("load_factor", {o: m.lf[i].mean() - truth["load_factor"][o] for i, o in enumerate(m.origins)},
        "max_abs", 0.02, "annual mean load factor per origin, absolute error")
    add("transfer_share", {o: m.tr[i].mean() - truth["transfer_share"][o] for i, o in enumerate(m.origins)},
        "max_abs", 0.02, "annual mean transfer share per origin, absolute error")
    add("alos", {n: m.alos[j].mean() / truth["alos"][n] - 1 for j, n in enumerate(m.nationalities)},
        "max_abs", 0.05, "annual mean length of stay per nationality, relative error")
    nas = par.non_air_share(l2, p)
    add("non_air_share", {n: nas[n] - truth["non_air_share_last12"][n] for n in l2.nationalities if n != "AE"},
        "median_abs", 0.15, "non-air share of guests (last 12 months) per nationality, absolute error (L2)")
    eff = l2.bridge["K"] * l2.bridge["m_hat"].mean(axis=1)[None, :]
    mix = bridge.mix_matrix(eff)
    add("bridge_diagonal", {o: mix[i, l2.nationalities.index(o)] - truth["mix_diagonal"][o]
                            for i, o in enumerate(l2.origins)},
        "median_abs", 0.15, "share of an origin's hotel guests holding that origin's nationality, absolute error")
    add("capture_rate", {o: eff[i, l2.nationalities.index(o)] / truth["effective_capture_diagonal"][o] - 1
                         for i, o in enumerate(l2.origins)},
        "median_abs", 0.25, "own-nationality hotel guests per terminating passenger (K diagonal), relative error")
    ae = nas.get("AE", np.nan)
    checks["domestic_is_non_air"] = {"statistic": "value", "value": float(ae), "tolerance": 0.999,
                                     "pass": bool(ae >= 0.999),
                                     "description": "UAE nationals are entirely non-air", "errors": {}}
    return {"note": "Parameter recovery on synthetic data with known ground truth. Evidence that the method "
                    "recovers known parameters; not evidence of real-world accuracy.",
            "checks": checks, "all_pass": all(c["pass"] for c in checks.values())}
