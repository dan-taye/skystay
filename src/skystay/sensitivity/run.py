"""`skystay sensitivity`: tornado, Sobol indices, route attractiveness, seasonal vulnerability, Planner Brief.

Writes reports/sensitivity.json, reports/planner_brief.{json,md} and figures.
"""

from __future__ import annotations

import json
import time

import numpy as np
import plotly.graph_objects as go
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from skystay import figures as fx
from skystay import paths, settings, train
from skystay.reconcile import regions
from skystay.scenario.engine import Engine
from skystay.scenario.levers import Frequency, NewRoute, Scenario
from skystay.sensitivity.fast import FACTORS, KEYS, FastEval, historical_sd

SOBOL_N = 2048
COLD_CANDIDATES = ["CA", "ZA", "KR", "ES", "NL", "TR", "MY", "NZ"]
LOCAL_LEVERS = ["capacity", "load_factor", "transfer_share", "capture", "alos", "events"]


def tornado(fe: FastEval, eng: Engine, sd: dict) -> dict:
    x0 = fe.neutral()
    base = fe.evaluate(x0)["nights"][0]
    base_tot = float(base.sum())
    top = [eng.nats[j] for j in np.argsort(-base.sum(axis=1))[:5]]
    a = eng.baseline.arrays
    w = a["S"]
    lf_mean = float((a["lf"] * w).sum() / w.sum())
    tr_mean = float((a["tr"] * w).sum() / w.sum())
    ev_mean = float(eng.event_days.mean())
    # step sizes in each factor's own unit for "+/-10%" and "+/-1 historical SD"
    steps = {
        "capacity": (0.10, sd["capacity"]),
        "load_factor": (10 * lf_mean, sd["load_factor"]),
        "transfer_share": (10 * tr_mean, sd["transfer_share"]),
        "capture": (0.10, sd["capture"]),
        "alos": (0.10, sd["alos"]),
        "events": (0.10 * ev_mean, sd["events"]),
    }
    rows = []
    for key in LOCAL_LEVERS:
        k = KEYS.index(key)
        out = {"lever": key, "label": FACTORS[k].label, "unit": FACTORS[k].unit}
        for tag, step in zip(("pct10", "sd1"), steps[key], strict=True):
            X = np.tile(x0, (2, 1))
            if FACTORS[k].neutral == 1.0:  # multiplicative factor
                X[0, k], X[1, k] = 1 - step, 1 + step
            else:
                X[0, k], X[1, k] = -step, step
            n = fe.evaluate(X)["nights"]
            tot = n.sum(axis=(1, 2)) - base_tot
            by = {m: [float(n[i, eng.nats.index(m)].sum() - base[eng.nats.index(m)].sum()) for i in (0, 1)]
                  for m in top}
            out[tag] = {"step": float(step), "low": float(tot[0]), "high": float(tot[1]),
                        "low_pct": float(tot[0] / base_tot), "high_pct": float(tot[1] / base_tot), "by_market": by}
        rows.append(out)
    rows.sort(key=lambda r: -(abs(r["pct10"]["high"]) + abs(r["pct10"]["low"])))
    return {"baseline_nights": base_tot, "top_markets": top, "levers": rows,
            "note": "Effect on annual hotel guest nights of moving one lever at a time, all markets together. "
                    "'+/-10%' moves the lever by 10% of its value; '+/-1 SD' by one historical year-to-year "
                    "standard deviation. The non-air baseline is not a lever."}


def sobol(fe: FastEval) -> dict:
    problem = {"num_vars": len(FACTORS), "names": KEYS, "bounds": [[f.lo, f.hi] for f in FACTORS]}
    X = sobol_sample.sample(problem, SOBOL_N, calc_second_order=False, seed=42)
    t = time.time()
    Y = fe.evaluate(X)["nights"].sum(axis=(1, 2))
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, conf_level=0.95, seed=42)
    rows = []
    for k, f in enumerate(FACTORS):
        rows.append({"factor": f.key, "label": f.label, "range": [f.lo, f.hi], "unit": f.unit,
                     "S1": float(Si["S1"][k]), "S1_ci95": float(Si["S1_conf"][k]),
                     "ST": float(Si["ST"][k]), "ST_ci95": float(Si["ST_conf"][k])})
    rows.sort(key=lambda r: -r["ST"])
    return {"N": SOBOL_N, "evaluations": int(len(X)), "seconds": round(time.time() - t, 1), "output": "annual guest nights",
            "y_mean": float(Y.mean()), "y_sd": float(Y.std()), "factors": rows,
            "method": "SALib Sobol sequence (Saltelli scheme), first- and total-order indices with 95% bootstrap CIs"}


def per_thousand(eng: Engine) -> dict:
    rows = []
    base_seats = eng.baseline.arrays["S"].sum(axis=1)
    for i, o in enumerate(eng.baseline.arrays["origins"]):
        if base_seats[i] <= 0:
            continue
        r = eng.run(Scenario(name=f"+10% {o}", levers=[Frequency(market=o, pct=10)]))
        ds = r.totals["seats"] - r.totals["baseline_seats"]
        if ds <= 0:
            continue
        di = r.totals["delta_guests_interval"]
        rows.append({"origin": o, "name": regions.label(o), "status": "served", "seats_added": ds,
                     "guests_per_1000_seats": 1000 * r.totals["delta_guests"] / ds,
                     "nights_per_1000_seats": 1000 * r.totals["delta_nights"] / ds,
                     "lo80": 1000 * di["lo80"] / ds, "hi80": 1000 * di["hi80"] / ds})
    for o in COLD_CANDIDATES:
        if o in eng.baseline.arrays["origins"]:
            continue
        r = eng.run(Scenario(name=f"new {o}", levers=[NewRoute(origin=o, weekly_freq=7, aircraft="B787-9")]))
        ds = r.totals["seats"] - r.totals["baseline_seats"]
        di = r.totals["delta_guests_interval"]
        rows.append({"origin": o, "name": regions.label(o), "status": "cold-start estimate", "seats_added": ds,
                     "guests_per_1000_seats": 1000 * r.totals["delta_guests"] / ds,
                     "nights_per_1000_seats": 1000 * r.totals["delta_nights"] / ds,
                     "lo80": 1000 * di["lo80"] / ds, "hi80": 1000 * di["hi80"] / ds})
    rows.sort(key=lambda r: -r["guests_per_1000_seats"])
    return {"rows": rows, "method": "served markets: +10% weekly frequency on all of the market's routes; "
            "candidate markets: a new daily B787-9 route (cold-start: region-average parameters, doubled "
            "uncertainty). Hotel guests across ALL nationalities per 1,000 added seats over the reference year; "
            "80% interval from the engine's Monte Carlo."}


def vulnerability(eng: Engine) -> dict:
    rows = []
    bl = eng.baseline
    for s in settings.seasons():
        r = eng.run(Scenario(name=f"-10% {s}", levers=[Frequency(market="ALL", pct=-10, season=s)]))
        mask = eng.season_mask(s)
        season_nights = float(bl.arrays["nights"][:, mask].sum())
        d = r.table.groupby("market")["delta_nights"].sum().sort_values()
        rows.append({"season": s, "delta_nights": r.totals["delta_nights"],
                     "delta_nights_interval": r.totals["delta_nights_interval"],
                     "delta_guests": r.totals["delta_guests"], "season_nights": season_nights,
                     "pct_of_season": r.totals["delta_nights"] / season_nights,
                     "most_affected": [{"market": k, "name": regions.label(k), "delta_nights": float(v)}
                                       for k, v in d.head(3).items()]})
    worst = min(rows, key=lambda r: r["pct_of_season"])
    return {"rows": rows, "most_vulnerable": worst["season"],
            "method": "10% fewer weekly flights on every route in one season; change in hotel guest nights"}


def brief(sens: dict, eng: Engine) -> dict:
    """Template-generated Planner Brief (no language model)."""
    syn = settings.is_synthetic()
    met = json.loads((paths.REPORTS / "metrics.json").read_text(encoding="utf-8"))
    model = json.loads((paths.REPORTS / "model.json").read_text(encoding="utf-8"))
    pt = sens["per_1000_seats"]["rows"]
    served = [r for r in pt if r["status"] == "served"]
    cold = [r for r in pt if r["status"] != "served"]
    sob = sens["sobol"]["factors"]
    vul = sens["vulnerability"]
    worst = next(r for r in vul["rows"] if r["season"] == vul["most_vulnerable"])
    ho = met["chosen"]["holdout"]["guests"]["total"]
    cov = met["chosen"]["coverage_holdout"]["guests"]
    nas = model["last12"]["non_air_share_total"]
    low = served[-3:]

    def f0(x):
        return f"{x:,.0f}"

    sections = {
        "best_routes": {
            "title": "Best routes to grow",
            "text": "Ranked by hotel guests gained per 1,000 extra seats: "
                    + "; ".join(f"{r['name']} {f0(r['guests_per_1000_seats'])} (80%: {f0(r['lo80'])} to "
                                f"{f0(r['hi80'])})" for r in served[:3])
                    + ". Lowest yield: " + ", ".join(f"{r['name']} {f0(r['guests_per_1000_seats'])}" for r in low)
                    + ". Seats into high-yield markets fill more hotel rooms than the same seats elsewhere.",
        },
        "new_markets": {
            "title": "New markets worth testing (cold-start estimates)",
            "text": "No service today, so these borrow region-average behaviour and carry wide intervals: "
                    + "; ".join(f"{r['name']} {f0(r['guests_per_1000_seats'])} guests per 1,000 seats (80%: "
                                f"{f0(r['lo80'])} to {f0(r['hi80'])})" for r in cold[:3])
                    + ". Treat them as hypotheses for a trial season, not forecasts.",
        },
        "seasonal_risk": {
            "title": "Seasonal risk",
            "text": f"A 10% capacity cut costs the most in {worst['season']}: {f0(-worst['delta_nights'])} guest "
                    f"nights ({-worst['pct_of_season']:.1%} of that season). Most exposed markets: "
                    + ", ".join(m["name"] for m in worst["most_affected"]) + ".",
        },
        "what_matters": {
            "title": "What moves hotel demand most",
            "text": f"Across plausible ranges of all levers, {sob[0]['label'].lower()} explains the largest share "
                    f"of the variation in annual guest nights (total-order Sobol index {sob[0]['ST']:.2f}), "
                    f"followed by {sob[1]['label'].lower()} ({sob[1]['ST']:.2f}) and {sob[2]['label'].lower()} "
                    f"({sob[2]['ST']:.2f}).",
        },
        "do_differently": {
            "title": "What a DCT planner should do differently",
            "bullets": [
                "Rank route-development incentives by hotel guests per 1,000 seats, not by seats or passengers.",
                f"Protect {worst['season']} capacity on "
                + ", ".join(m["name"] for m in worst["most_affected"][:2])
                + " first: that is where a capacity loss hurts hotels most.",
                f"Plan non-air demand separately: about {nas:.0%} of hotel guests in the latest year do not depend "
                "on flights into Abu Dhabi, so airline deals cannot move them.",
                "Agree one shared number with airports and airlines: use the simulator's central estimate for "
                "targets and its 80% range for staffing and inventory buffers.",
                "Test new markets with a trial season and re-estimate after 6 to 12 months of data.",
            ],
        },
        "confidence": {
            "title": "Confidence notes",
            "text": (f"Back-tested on held-out months: WMAPE {ho['wmape']:.1%} on monthly hotel guests by market in "
                     f"the final holdout; 80% intervals covered {cov['0.8']:.0%} of outcomes and 95% intervals "
                     f"{cov['0.95']:.0%}. Cold-start and out-of-range scenarios carry wider intervals.")
                    + (" All figures come from SYNTHETIC DEMO DATA with known ground truth and say nothing about "
                       "real Abu Dhabi demand." if syn else ""),
        },
    }
    return {"data_mode": settings.data_mode(), "sections": sections}


def brief_markdown(b: dict) -> str:
    s = b["sections"]
    L = ["# Planner Brief", ""]
    if b["data_mode"] == "synthetic":
        L += ["> **SYNTHETIC DEMO DATA.** Generated from the synthetic stand-in; not about real Abu Dhabi demand.", ""]
    for k in ("best_routes", "new_markets", "seasonal_risk", "what_matters"):
        L += [f"## {s[k]['title']}", "", s[k]["text"], ""]
    L += [f"## {s['do_differently']['title']}", ""] + [f"- {x}" for x in s["do_differently"]["bullets"]] + [""]
    L += [f"## {s['confidence']['title']}", "", s["confidence"]["text"], ""]
    return "\n".join(L)


# ------------------------------------------------------------------ figures
def fig_tornado(t: dict, which: str = "pct10") -> go.Figure:
    rows = list(reversed(t["levers"]))
    labels = [r["label"] for r in rows]
    lo = [r[which]["low"] for r in rows]
    hi = [r[which]["high"] for r in rows]
    tag = "+/-10%" if which == "pct10" else "+/-1 historical SD"
    fig = go.Figure()
    fig.add_bar(y=labels, x=lo, orientation="h", name="Lever lowered", marker_color=fx.DIV_NEG,
                hovertemplate="%{y}<br>Lever lowered: %{x:,.0f} nights<extra></extra>")
    fig.add_bar(y=labels, x=hi, orientation="h", name="Lever raised", marker_color=fx.DIV_POS,
                hovertemplate="%{y}<br>Lever raised: %{x:+,.0f} nights<extra></extra>")
    fig.update_layout(barmode="overlay", bargap=0.35)
    fx.layout(fig, f"Tornado: change in annual guest nights, one lever at a time ({tag})", height=430)
    fig.update_xaxes(title="Change in annual hotel guest nights", tickformat="+,.0f", showgrid=True,
                     gridcolor=fx.GRID, zeroline=True, zerolinecolor=fx.AXIS)
    return fig


def fig_sobol(s: dict) -> go.Figure:
    rows = list(reversed(s["factors"]))
    labels = [r["label"] for r in rows]
    fig = go.Figure()
    fig.add_bar(y=labels, x=[r["S1"] for r in rows], orientation="h", name="First-order (alone)",
                marker_color=fx.SERIES[1], error_x={"type": "data", "array": [r["S1_ci95"] for r in rows],
                                                     "color": fx.INK_2, "thickness": 1})
    fig.add_bar(y=labels, x=[r["ST"] for r in rows], orientation="h", name="Total-order (incl. interactions)",
                marker_color=fx.SERIES[0], error_x={"type": "data", "array": [r["ST_ci95"] for r in rows],
                                                     "color": fx.INK_2, "thickness": 1})
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fx.layout(fig, "Global sensitivity (Sobol): share of variance in annual guest nights", height=470)
    fig.update_xaxes(title="Sobol index (0 to 1), 95% CI", range=[0, 1], showgrid=True, gridcolor=fx.GRID)
    return fig


def fig_per1000(p: dict) -> go.Figure:
    rows = list(reversed(p["rows"]))
    fig = go.Figure()
    for status, color, name in (("served", fx.SERIES[0], "Served market (+10% frequency)"),
                                ("cold-start estimate", fx.SERIES[1], "New market (cold-start estimate)")):
        rr = [r for r in rows if r["status"] == status]
        fig.add_bar(y=[r["name"] for r in rr], x=[r["guests_per_1000_seats"] for r in rr], orientation="h",
                    name=name, marker_color=color,
                    error_x={"type": "data", "symmetric": False,
                             "array": [r["hi80"] - r["guests_per_1000_seats"] for r in rr],
                             "arrayminus": [r["guests_per_1000_seats"] - r["lo80"] for r in rr],
                             "color": fx.INK_2, "thickness": 1},
                    hovertemplate="%{y}: %{x:,.0f} guests per 1,000 seats<extra></extra>")
    fx.layout(fig, "Route attractiveness: hotel guests per 1,000 added seats (80% interval)", height=820)
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(categoryorder="array", categoryarray=[r["name"] for r in rows])
    fig.update_xaxes(title="Hotel guests per 1,000 added seats", showgrid=True, gridcolor=fx.GRID)
    return fig


def fig_vulnerability(v: dict) -> go.Figure:
    rows = v["rows"]
    vals = [-r["pct_of_season"] * 100 for r in rows]
    fig = go.Figure(go.Bar(x=[r["season"] for r in rows], y=vals, marker_color=fx.DIV_NEG,
                           text=[f"-{x:.1f}%  ({-r['delta_nights']:,.0f} nights)" for x, r in zip(vals, rows, strict=True)],
                           textposition="outside", textfont={"color": fx.INK_2}))
    fx.layout(fig, "Seasonal vulnerability: guest nights lost if capacity is cut 10% in one season", height=420,
              legend=False)
    fig.update_yaxes(title="Guest nights lost (% of the season)", ticksuffix="%", rangemode="tozero")
    return fig


def run() -> dict:
    paths.ensure_dirs()
    t0 = time.time()
    eng = Engine(train.load())
    fe = FastEval(eng)
    sd = historical_sd(eng)
    out = {"data_mode": settings.data_mode(), "reference_year": f"{eng.months[0]:%Y-%m}..{eng.months[-1]:%Y-%m}",
           "historical_sd": sd}
    out["tornado"] = tornado(fe, eng, sd)
    print(f"tornado done ({time.time() - t0:.0f}s)")
    out["sobol"] = sobol(fe)
    print(f"sobol done: {out['sobol']['evaluations']} evaluations in {out['sobol']['seconds']}s")
    out["per_1000_seats"] = per_thousand(eng)
    out["vulnerability"] = vulnerability(eng)
    out["planner_brief"] = brief(out, eng)
    (paths.REPORTS / "sensitivity.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    (paths.REPORTS / "planner_brief.md").write_text(brief_markdown(out["planner_brief"]), encoding="utf-8")
    fx.save_png({
        "tornado": fig_tornado(out["tornado"]),
        "tornado_sd": fig_tornado(out["tornado"], "sd1"),
        "sobol": fig_sobol(out["sobol"]),
        "guests_per_1000_seats": fig_per1000(out["per_1000_seats"]),
        "seasonal_vulnerability": fig_vulnerability(out["vulnerability"]),
    })
    from skystay import docs

    docs.write_all()
    top = out["sobol"]["factors"][0]
    print(f"top lever by total-order Sobol: {top['label']} (ST {top['ST']:.2f}); total {time.time() - t0:.0f}s")
    return out
