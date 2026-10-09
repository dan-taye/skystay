"""`skystay backtest`: rolling-origin back-test of every model at every reconciliation level.

For each fold, everything (ratios, shrinkage, calendar uplifts, bridge,
baseline, econometric and ML layers) is refit on months strictly before the
cutoff, then the next 12 months are predicted from the schedule (seats) and
the calendars. The (reconciliation level, model) pair with the lowest mean
cross-validated WMAPE on hotel guests is selected; the final 12-month holdout
is then scored once for every model and reported.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from skystay import paths, settings
from skystay.backtest import conformal, metrics
from skystay.backtest.splits import Fold, make_folds
from skystay.chain import panel
from skystay.chain import parametric as par
from skystay.ml import econometric, features, gbm, naive
from skystay.reconcile import regions

MODELS = ("seasonal_naive", "parametric", "econometric", "ml_only", "hybrid")
MODEL_LABELS = {
    "seasonal_naive": "Seasonal naive",
    "parametric": "Parametric chain",
    "econometric": "Econometric",
    "ml_only": "ML only",
    "hybrid": "Hybrid (chain x ML)",
}
LEVEL_LABELS = {"L0": "L0 identity", "L1": "L1 regional", "L2": "L2 learned bridge"}
PRED_FILE = "backtest_predictions.parquet"

TRAIN_LOG: list[dict] = []  # every training frame's last month vs its fold cutoff (inspected by the leakage test)


class LeakageError(AssertionError):
    pass


def guard(name: str, fold: Fold, months: pd.Series | pd.DatetimeIndex) -> None:
    last = pd.Timestamp(pd.Series(months).max())
    TRAIN_LOG.append({"fold": fold.name, "what": name, "last_month": last, "cutoff": fold.cutoff})
    if last >= fold.cutoff:
        raise LeakageError(f"{fold.name}/{name}: training data reaches {last:%Y-%m} >= cutoff {fold.cutoff:%Y-%m}")


def run_fold(p: panel.Panels, fold: Fold, levels: tuple[str, ...] = par.LEVELS) -> pd.DataFrame:
    ptr = panel.restrict(p, fold.cutoff)
    guard("origin_month", fold, ptr.origin_month["month"])
    guard("nat_month", fold, ptr.nat_month["month"])
    actual = p.nat_month[p.nat_month["month"].isin(fold.test)][["nationality", "month", "guests", "nights"]]
    out = []
    sn = actual[["nationality", "month"]].copy()
    sn["pred_guests"] = naive.predict(ptr.nat_month, sn, "guests")
    sn["pred_nights"] = naive.predict(ptr.nat_month, sn, "nights")
    for level in levels:
        m = par.fit(ptr, level)
        tr_seats = features.frame(ptr, m, ptr.months, use_actual_supply=False)
        tr_act = features.frame(ptr, m, ptr.months, use_actual_supply=True)
        guard("ml_train", fold, tr_seats.loc[tr_seats["guests"].notna(), "month"])
        te = features.frame(p, m, fold.test, use_actual_supply=False)
        guard("lag12_features", fold, te.loc[te["guests_lag12"].notna(), "month"] - pd.DateOffset(months=12))
        alos = m.alos[[m.nat_index()[n] for n in te["nationality"]], te["moy"].to_numpy() - 1]
        snm = te[["nationality", "month"]].merge(sn, on=["nationality", "month"], how="left")
        preds = {
            "seasonal_naive": (snm["pred_guests"].to_numpy(), snm["pred_nights"].to_numpy()),
            "parametric": (te["guests_pred"].to_numpy(), te["nights_pred"].to_numpy()),
        }
        em = econometric.fit(tr_act)
        g = econometric.predict(em, te)
        preds["econometric"] = (g, g * alos)
        for target in ("ml_only", "hybrid"):
            gm = gbm.fit(tr_seats, target)
            g = gbm.predict(gm, te)
            preds[target] = (g, g * alos)
        for model, (pg, pn) in preds.items():
            df = te[["nationality", "month"]].copy()
            df["fold"], df["level"], df["model"] = fold.name, level, model
            df["pred_guests"], df["pred_nights"] = pg, pn
            out.append(df)
    res = pd.concat(out, ignore_index=True).merge(actual, on=["nationality", "month"], how="left")
    res["holdout"] = fold.holdout
    return res


def _scores(df: pd.DataFrame, scales: dict[str, pd.Series]) -> dict:
    return {
        "guests": metrics.block(df, "guests", "pred_guests", scales.get("guests")),
        "nights": metrics.block(df, "nights", "pred_nights", scales.get("nights")),
    }


def _table(preds: pd.DataFrame, folds: list[Fold], p: panel.Panels) -> dict:
    """Metrics per (level, model): per fold, CV mean, and holdout."""
    out: dict = {}
    for (level, model), g in preds.groupby(["level", "model"]):
        entry: dict = {"by_fold": {}}
        for f in folds:
            gf = g[g["fold"] == f.name]
            hist = p.nat_month[p.nat_month["month"] < f.cutoff]
            scales = {"guests": metrics.snaive_scale(hist, "guests"), "nights": metrics.snaive_scale(hist, "nights")}
            entry["by_fold"][f.name] = _scores(gf, scales)
        cv = [entry["by_fold"][f.name] for f in folds if not f.holdout]
        for k in ("guests", "nights"):
            entry[f"cv_{k}"] = {
                "wmape_mean": float(np.mean([c[k]["wmape"] for c in cv])),
                "bias_mean": float(np.mean([c[k]["bias"] for c in cv])),
                "mase_mean": float(np.mean([c[k]["mase"] for c in cv])),
            }
        entry["holdout"] = entry["by_fold"]["holdout"]
        out.setdefault(level, {})[model] = entry
    return out


def _season(df: pd.DataFrame) -> pd.Series:
    return df["month"].dt.month.map(settings.season_of_month)


def _detail(df: pd.DataFrame) -> dict:
    d = df.assign(season=_season(df))
    return {
        "guests": {"total": metrics.block(d, "guests", "pred_guests"),
                   "by_season": metrics.grouped(d, "season", "guests", "pred_guests"),
                   "by_market": metrics.grouped(d, "nationality", "guests", "pred_guests")},
        "nights": {"total": metrics.block(d, "nights", "pred_nights"),
                   "by_season": metrics.grouped(d, "season", "nights", "pred_nights"),
                   "by_market": metrics.grouped(d, "nationality", "nights", "pred_nights")},
    }


def _worst_markets(df: pd.DataFrame, p: panel.Panels, quality: dict, k: int = 5) -> list[dict]:
    """The k markets with the highest holdout WMAPE, each with a fact-based explanation."""
    reg = regions.region_of()
    by = metrics.grouped(df, "nationality", "guests", "pred_guests")
    svc = {s["origin"]: s for s in quality.get("service_starts_and_stops", [])}
    served = set(p.origin_month.loc[p.origin_month["seats"] > 0, "origin"])
    missing = {m["nationality"]: m["missing"] for m in quality.get("missing_hotel_months", [])}
    total = df["guests"].sum()
    rows = []
    for n, v in sorted(by.items(), key=lambda kv: -kv[1]["wmape"])[:k]:
        g = df[df["nationality"] == n]
        share = g["guests"].sum() / total if total else 0
        reasons = []
        if share < 0.01:
            reasons.append(f"small market ({share:.1%} of guests), so modest absolute errors are large in % terms")
        if n not in served:
            reasons.append("no direct route: these guests arrive via hubs, neighbouring markets or other gateways, "
                           "which the model can only infer indirectly")
        if n in svc:
            s = svc[n]
            reasons.append(f"route history is short or interrupted (served {s['first']} to {s['last']})")
        if n in missing:
            reasons.append(f"hotel data missing for {', '.join(missing[n])}")
        b = v["bias"]
        if abs(b) > 0.5 * v["wmape"]:
            reasons.append(f"systematic {'over' if b > 0 else 'under'}-prediction (bias {b:+.1%})")
        if not reasons:
            reasons.append("no structural cause identified; error is within normal month-to-month noise")
        rows.append({"nationality": n, "name": regions.label(n), "region": reg.get(n, "OTH"), "wmape": v["wmape"],
                     "bias": b, "guest_share": float(share), "explanation": "; ".join(reasons)})
    return rows


def run() -> dict:
    paths.ensure_dirs()
    TRAIN_LOG.clear()
    s = settings.load_settings()
    p = panel.build()
    bt = s["backtest"]
    folds = make_folds(p.months, bt["n_folds"], bt["horizon_months"], bt["holdout_months"])
    print("folds:", [f.describe()["name"] + " " + f.describe()["test_first"] for f in folds])
    preds = pd.concat([run_fold(p, f) for f in folds], ignore_index=True)
    preds.to_parquet(paths.PROCESSED / PRED_FILE, index=False)

    table = _table(preds, folds, p)
    cv_rank = sorted(
        ((lv, md, table[lv][md]["cv_guests"]["wmape_mean"]) for lv in table for md in table[lv]),
        key=lambda x: x[2],
    )
    level, model, best = cv_rank[0]
    print(f"selected: {level} / {model} (mean CV WMAPE guests {best:.4f})")

    # conformal calibration on out-of-fold residuals of the selected pair
    reg = regions.region_of()
    chosen = preds[(preds["level"] == level) & (preds["model"] == model)].copy()
    chosen["region"] = chosen["nationality"].map(reg)
    cv = chosen[~chosen["holdout"]]
    ho = chosen[chosen["holdout"]].copy()
    q = {}
    cover = {}
    for k in ("guests", "nights"):
        r = cv[(cv[k] > 0) & (cv[f"pred_{k}"] > 0)]
        resid = pd.DataFrame({"nationality": r["nationality"], "region": r["region"],
                              "r": np.log(r[k] / r[f"pred_{k}"])})
        q[k] = conformal.calibrate(resid)
        cover[k] = {}
        for lv in conformal.LEVELS:
            lo, hi = conformal.bands(q[k], ho["nationality"], ho[f"pred_{k}"].to_numpy(), lv)
            ho[f"{k}_lo{int(lv * 100)}"], ho[f"{k}_hi{int(lv * 100)}"] = lo, hi
            cover[k][str(lv)] = conformal.coverage(ho[k].to_numpy(), lo, hi)
    # bands on CV folds too (for charts): calibrated on the other folds would be stricter; we label them in-sample
    for k in ("guests",):
        for lv in conformal.LEVELS:
            lo, hi = conformal.bands(q[k], cv["nationality"], cv[f"pred_{k}"].to_numpy(), lv)
            chosen.loc[cv.index, f"{k}_lo{int(lv * 100)}"] = lo
            chosen.loc[cv.index, f"{k}_hi{int(lv * 100)}"] = hi
    for c in [c for c in ho.columns if c.endswith(("lo80", "hi80", "lo95", "hi95"))]:
        chosen.loc[ho.index, c] = ho[c]
    chosen.to_parquet(paths.PROCESSED / "backtest_chosen.parquet", index=False)

    quality = json.loads((paths.REPORTS / "data_quality.json").read_text(encoding="utf-8"))
    recon = {lv: {md: {"cv_wmape": table[lv][md]["cv_guests"]["wmape_mean"],
                       "holdout_wmape": table[lv][md]["holdout"]["guests"]["wmape"]}
                  for md in ("parametric", "hybrid", "econometric", "ml_only")} for lv in table}
    recon_choice = min(table, key=lambda lv: table[lv]["parametric"]["cv_guests"]["wmape_mean"])
    out = {
        "data_mode": settings.data_mode(),
        "title_note": ("on synthetic data with known ground truth" if settings.is_synthetic()
                       else "on real data"),
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "setup": {
            "target": "monthly hotel guests (and guest nights) by nationality",
            "inputs_at_forecast_time": "flight schedule (seats) for the test months, event and holiday calendars; "
                                       "load factor, transfer share and all conversion parameters estimated on "
                                       "training months only",
            "horizon_months": bt["horizon_months"],
            "markets": len(p.nationalities),
        },
        "folds": [f.describe() for f in folds],
        "selection": {"level": level, "model": model, "criterion": "lowest mean CV WMAPE on hotel guests",
                      "cv_wmape_guests": best, "level_label": LEVEL_LABELS[level], "model_label": MODEL_LABELS[model],
                      "ranking": [{"level": a, "model": b, "cv_wmape": c} for a, b, c in cv_rank]},
        "reconciliation": {"by_level": recon, "best_level_for_parametric_chain": recon_choice},
        "table": table,
        "chosen": {
            "cv": _detail(cv),
            "holdout": _detail(ho),
            "coverage_holdout": cover,
            "conformal": q,
            "worst_markets_holdout": _worst_markets(ho, p, quality),
        },
    }
    (paths.REPORTS / "metrics.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    from skystay.backtest import report

    report.write(out, preds, chosen, p)
    h = out["chosen"]["holdout"]["guests"]["total"]
    print(f"holdout WMAPE guests {h['wmape']:.4f}, bias {h['bias']:+.4f}; coverage 80% "
          f"{cover['guests']['0.8']:.2f}, 95% {cover['guests']['0.95']:.2f}")
    return out
