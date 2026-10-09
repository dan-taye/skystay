"""Generated documentation: the README results block and docs/MODEL_CARD.md.

Every number comes from reports/*.json. Called at the end of `skystay sensitivity`
(and by the deck build), so the docs never drift from the pipeline outputs.
"""

from __future__ import annotations

import json
import re

from skystay import paths, settings

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _load(name: str) -> dict:
    f = paths.REPORTS / f"{name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _p(x, d=1) -> str:
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def results_block() -> str:
    met, mod, sens, rec = _load("metrics"), _load("model"), _load("sensitivity"), _load("recovery")
    if not met:
        return "_Run `skystay all` to generate results._"
    syn = met["data_mode"] == "synthetic"
    sel = met["selection"]
    lv = sel["level"]
    t = met["table"][lv]
    ho = met["chosen"]["holdout"]
    cov = met["chosen"]["coverage_holdout"]
    L = []
    if syn:
        L += ["> **SYNTHETIC DEMO DATA.** These results are on synthetic data with known ground truth. They show the "
              "method works; they are not real-world accuracy.", ""]
    L += [f"Selected by rolling-origin back-test: **{sel['level_label']} + {sel['model_label']}**.", "",
          "| Model | CV WMAPE (guests) | Holdout WMAPE (guests) | Holdout WMAPE (nights) |", "| :- | -: | -: | -: |"]
    labels = {"seasonal_naive": "Seasonal naive", "parametric": "Parametric chain", "econometric": "Econometric",
              "ml_only": "ML only", "hybrid": "Hybrid (chain x ML)"}
    for m, lab in labels.items():
        e = t[m]
        L.append(f"| {lab} | {_p(e['cv_guests']['wmape_mean'])} | {_p(e['holdout']['guests']['wmape'])} | "
                 f"{_p(e['holdout']['nights']['wmape'])} |")
    L += ["", f"- Holdout ({met['folds'][-1]['test_first']} to {met['folds'][-1]['test_last']}): WMAPE "
          f"{_p(ho['guests']['total']['wmape'])} on monthly hotel guests by market, bias "
          f"{ho['guests']['total']['bias']:+.1%}.",
          f"- Conformal intervals on the holdout: 80% covered {_p(cov['guests']['0.8'], 0)}, 95% covered "
          f"{_p(cov['guests']['0.95'], 0)}."]
    if mod:
        L.append(f"- Non-air share of hotel guests (latest year): {_p(mod['last12']['non_air_share_total'], 0)}.")
    if rec:
        L.append(f"- Parameter recovery on synthetic ground truth: "
                 f"{sum(c['pass'] for c in rec['checks'].values())}/{len(rec['checks'])} checks pass.")
    if sens:
        s = sens["sobol"]["factors"]
        best = [r for r in sens["per_1000_seats"]["rows"] if r["status"] == "served"][:3]
        L.append(f"- Biggest drivers of annual guest nights (Sobol total-order): {s[0]['label']} ({s[0]['ST']:.2f}), "
                 f"{s[1]['label']} ({s[1]['ST']:.2f}), {s[2]['label']} ({s[2]['ST']:.2f}).")
        L.append("- Highest hotel guests per 1,000 added seats: "
                 + ", ".join(f"{r['name']} ({r['guests_per_1000_seats']:,.0f})" for r in best) + ".")
    L += ["", "Full details: `reports/backtest.md`, `reports/metrics.json`, `reports/sensitivity.json`, "
          "`reports/planner_brief.md`."]
    return "\n".join(L)


def write_readme() -> None:
    f = paths.ROOT / "README.md"
    if not f.exists():
        return
    s = f.read_text(encoding="utf-8")
    if START in s and END in s:
        s = re.sub(re.escape(START) + r".*?" + re.escape(END), f"{START}\n{results_block()}\n{END}", s, flags=re.S)
        f.write_text(s, encoding="utf-8")


def write_model_card() -> None:
    met, mod, rec, dq = _load("metrics"), _load("model"), _load("recovery"), _load("data_quality")
    if not met:
        return
    syn = met["data_mode"] == "synthetic"
    sel = met["selection"]
    e = mod.get("econometric", {}).get("elasticity", {})
    L = [f"# Model card: {settings.product_name()} flight-to-hotel demand model", ""]
    if syn:
        L += ["> **SYNTHETIC DEMO DATA.** Trained and evaluated on a seeded synthetic stand-in for real tourism data.", ""]
    L += [
        "## Intended use",
        "Scenario planning for Abu Dhabi tourism: estimate how changes in air capacity and conversion assumptions "
        "move hotel guests and guest nights by source market and month, with uncertainty. For planners at DCT and "
        "partners. **Not** for: forecasting beyond the reference year, individual-level decisions, or any use of "
        "passenger-level data (none is used).", "",
        "## Model",
        f"- Selected: **{sel['level_label']} + {sel['model_label']}** (lowest mean cross-validated WMAPE).",
        "- Layer 1 (transparent chain): seats x load factor x (1 - transfer share) = terminating passengers; a "
        "learned origin-to-nationality bridge K (non-negative, ridge toward a regional prior) converts them to "
        "air-attributable hotel guests by nationality, with a seasonal capture shape and pooled holiday/event "
        "uplifts; a non-air baseline (DCT seasonality profile, ridge-penalised trend) is added; guest nights = "
        "guests x length of stay. Empirical-Bayes shrinkage by month of year; cold-start from region averages.",
        "- Layer 2 (benchmark): OLS of log guests on log air visitors with month and market fixed effects, holiday "
        f"and event terms, HC3 errors. Elasticity {e.get('coef', float('nan')):.2f} (95% CI "
        f"{e.get('ci95', [float('nan')] * 2)[0]:.2f} to {e.get('ci95', [float('nan')] * 2)[1]:.2f}).",
        "- Layer 3: LightGBM on the log residual of Layer 1 (hybrid) with a monotone-increasing constraint on air "
        "visitors; features: market, month, holiday days, event days, trend, 12-month lag.",
        "- Intervals: split-conformal on out-of-fold log residuals (per market, region-pooled when thin), combined "
        "in scenarios with Monte Carlo over parameter uncertainty.", "",
        "## Training data",
        f"- Window {dq.get('overlap_window', {}).get('first')} to {dq.get('overlap_window', {}).get('last')} "
        f"({dq.get('overlap_window', {}).get('n_months')} months); {met['setup']['markets']} guest nationalities; "
        "aggregated monthly data only (see `docs/DATA_NOTES.md`).", "",
        "## Evaluation (" + met["title_note"] + ")", "",
        results_block(), "",
        "## Parameter recovery" + (" (synthetic only)" if syn else ""), "",
    ]
    if rec:
        L += ["| Check | Statistic | Value | Tolerance | Pass |", "| :- | :- | -: | -: | :- |"]
        for k, v in rec["checks"].items():
            L.append(f"| {k} | {v['statistic']} | {v['value']:.3f} | {v['tolerance']} | {'yes' if v['pass'] else 'NO'} |")
    L += ["", "## Limitations and ethical considerations",
          "See `docs/LIMITATIONS.md`. Key points: synthetic data until real data is available; air vs non-air demand "
          "only weakly identified; returning residents folded into capture; new routes are cold-start estimates "
          "without diversion; intervals calibrated on three folds. Aggregated data only; no personal data.", "",
          "## Reproducibility",
          "`skystay all` regenerates every number from raw files (seed 42). Tests: `skystay test`."]
    (paths.DOCS / "MODEL_CARD.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_all() -> None:
    write_readme()
    write_model_card()
