"""reports/backtest.md and back-test figures (all numbers from the runner's output)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from skystay import figures as fx
from skystay import paths, settings
from skystay.backtest.runner import LEVEL_LABELS, MODEL_LABELS, MODELS
from skystay.reconcile import regions

MODEL_ORDER = list(MODELS)


def _pct(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def _suffix(out: dict) -> str:
    return f" ({out['title_note']})"


def fig_top10(chosen: pd.DataFrame, out: dict) -> go.Figure:
    top = (chosen.groupby("nationality")["guests"].sum().sort_values(ascending=False).head(10).index.tolist())
    fig = make_subplots(rows=2, cols=5, subplot_titles=[regions.label(n) for n in top], shared_xaxes=False,
                        horizontal_spacing=0.05, vertical_spacing=0.17)
    ho_start = pd.Timestamp(next(f for f in out["folds"] if f["holdout"])["cutoff"])
    for i, n in enumerate(top):
        r, c = i // 5 + 1, i % 5 + 1
        g = chosen[chosen["nationality"] == n].sort_values("month")
        fig.add_trace(go.Scatter(mode="lines", x=g["month"], y=g["guests_hi80"], line={"width": 0}, showlegend=False,
                                 hoverinfo="skip"), r, c)
        fig.add_trace(go.Scatter(mode="lines", x=g["month"], y=g["guests_lo80"], line={"width": 0}, fill="tonexty",
                                 fillcolor=fx.BAND, name="80% interval", showlegend=i == 0, hoverinfo="skip"), r, c)
        fig.add_trace(go.Scatter(mode="lines", x=g["month"], y=g["pred_guests"], line={"color": fx.SERIES[0], "width": 2},
                                 name="Predicted", showlegend=i == 0,
                                 hovertemplate="%{x|%b %Y}<br>Predicted %{y:,.0f}<extra></extra>"), r, c)
        fig.add_trace(go.Scatter(x=g["month"], y=g["guests"], line={"color": fx.INK, "width": 1.5},
                                 mode="lines", name="Actual", showlegend=i == 0,
                                 hovertemplate="%{x|%b %Y}<br>Actual %{y:,.0f}<extra></extra>"), r, c)
        fig.add_vline(x=ho_start, line={"color": fx.AXIS, "dash": "dot", "width": 1}, row=r, col=c)
    fx.layout(fig, f"Hotel guests: actual vs back-test predictions, top 10 markets{_suffix(out)}", height=660)
    fig.update_layout(margin={"t": 130, "b": 70}, legend={"y": 1.075})
    fig.update_yaxes(tickformat="~s")
    fig.update_annotations(font={"size": 12, "color": fx.INK_2})
    fig.add_annotation(text="Dotted line: start of the untouched 2025 holdout. Earlier years: rolling-origin folds.",
                       xref="paper", yref="paper", x=0, y=-0.1, showarrow=False, font={"size": 11, "color": fx.MUTED},
                       xanchor="left")
    return fig


def fig_models(out: dict) -> go.Figure:
    lv = out["selection"]["level"]
    t = out["table"][lv]
    names = [MODEL_LABELS[m] for m in MODEL_ORDER]
    cv = [t[m]["cv_guests"]["wmape_mean"] * 100 for m in MODEL_ORDER]
    ho = [t[m]["holdout"]["guests"]["wmape"] * 100 for m in MODEL_ORDER]
    fig = go.Figure()
    fig.add_bar(y=names, x=cv, orientation="h", name="Cross-validation (mean of 3 folds)", marker_color=fx.SERIES[0],
                text=[f"{v:.1f}%" for v in cv], textposition="outside", textfont={"color": fx.INK_2})
    fig.add_bar(y=names, x=ho, orientation="h", name="2025 holdout", marker_color=fx.SERIES[1],
                text=[f"{v:.1f}%" for v in ho], textposition="outside", textfont={"color": fx.INK_2})
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fx.layout(fig, f"WMAPE on hotel guests by model, {LEVEL_LABELS[lv]} (lower is better){_suffix(out)}", height=420)
    fig.update_xaxes(title="WMAPE (%)", tickformat=".0f", showgrid=True, gridcolor=fx.GRID)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def fig_reconciliation(out: dict) -> go.Figure:
    rec = out["reconciliation"]["by_level"]
    levels = [lv for lv in ("L0", "L1", "L2") if lv in rec]
    fig = go.Figure()
    for k, (md, label) in enumerate((("parametric", "Parametric chain"), ("hybrid", "Hybrid (chain x ML)"))):
        vals = [rec[lv][md]["cv_wmape"] * 100 for lv in levels]
        fig.add_bar(x=[LEVEL_LABELS[lv] for lv in levels], y=vals, name=label, marker_color=fx.SERIES[k],
                    text=[f"{v:.1f}%" for v in vals], textposition="outside", textfont={"color": fx.INK_2})
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fx.layout(fig, f"Country reconciliation: cross-validated WMAPE on guests{_suffix(out)}", height=420)
    fig.update_yaxes(title="WMAPE (%)", tickformat=".0f")
    return fig


def fig_seasons(out: dict) -> go.Figure:
    seasons = list(settings.seasons().keys())
    fig = go.Figure()
    for k, (src, label) in enumerate((("cv", "Cross-validation folds"), ("holdout", "2025 holdout"))):
        bs = out["chosen"][src]["guests"]["by_season"]
        vals = [bs[s]["wmape"] * 100 for s in seasons]
        fig.add_bar(x=seasons, y=vals, name=label, marker_color=fx.SERIES[k], text=[f"{v:.1f}%" for v in vals],
                    textposition="outside", textfont={"color": fx.INK_2})
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fx.layout(fig, f"Selected model: WMAPE on guests by season{_suffix(out)}", height=400)
    fig.update_yaxes(title="WMAPE (%)", tickformat=".0f")
    return fig


def fig_markets(out: dict) -> go.Figure:
    bm = out["chosen"]["holdout"]["guests"]["by_market"]
    items = sorted(bm.items(), key=lambda kv: kv[1]["wmape"])
    names = [f"{regions.label(k)}" for k, _ in items]
    vals = [v["wmape"] * 100 for _, v in items]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=fx.SERIES[0],
                           text=[f"{v:.1f}%" for v in vals], textposition="outside", textfont={"color": fx.INK_2},
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fx.layout(fig, f"Selected model: 2025 holdout WMAPE by market{_suffix(out)}", height=760, legend=False)
    fig.update_xaxes(title="WMAPE (%)", tickformat=".0f", showgrid=True, gridcolor=fx.GRID)
    return fig


def write(out: dict, preds: pd.DataFrame, chosen: pd.DataFrame, p) -> None:
    figs = {
        "backtest_top10": fig_top10(chosen, out),
        "wmape_models": fig_models(out),
        "wmape_reconciliation": fig_reconciliation(out),
        "wmape_seasons": fig_seasons(out),
        "wmape_markets": fig_markets(out),
    }
    fx.save_png(figs)
    (paths.REPORTS / "backtest.md").write_text(markdown(out), encoding="utf-8")


def markdown(out: dict) -> str:
    syn = out["data_mode"] == "synthetic"
    sel = out["selection"]
    lv = sel["level"]
    L = [f"# Back-test results{_suffix(out)}", ""]
    if syn:
        L += ["> **SYNTHETIC DEMO DATA.** Every number below measures how well the method reproduces a synthetic "
              "world with known ground truth. None of it is a claim about real Abu Dhabi accuracy.", ""]
    L += ["## Set-up", "",
          f"- Target: {out['setup']['target']} ({out['setup']['markets']} markets).",
          f"- At forecast time the model sees: {out['setup']['inputs_at_forecast_time']}.",
          "- Rolling-origin folds with an expanding window and a 12-month horizon; everything is refit per fold on "
          "months before the cutoff (enforced by `tests/test_leakage.py`).", ""]
    L += ["| Fold | Training months | Test window |", "| :- | -: | :- |"]
    for f in out["folds"]:
        L.append(f"| {f['name']}{' (final holdout, scored once)' if f['holdout'] else ''} | {f['train_months']} | "
                 f"{f['test_first']} to {f['test_last']} |")
    L += ["", "## Selected model", "",
          f"**{sel['level_label']} + {sel['model_label']}**, chosen by {sel['criterion']} "
          f"({_pct(sel['cv_wmape_guests'])}).", ""]
    t = out["table"][lv]
    L += [f"## WMAPE against every baseline ({LEVEL_LABELS[lv]})", "",
          "| Model | CV WMAPE guests | CV WMAPE nights | CV bias | CV MASE | Holdout WMAPE guests | Holdout WMAPE "
          "nights | Holdout bias |", "| :- | -: | -: | -: | -: | -: | -: | -: |"]
    for m in MODEL_ORDER:
        e = t[m]
        L.append(f"| {MODEL_LABELS[m]} | {_pct(e['cv_guests']['wmape_mean'])} | {_pct(e['cv_nights']['wmape_mean'])} "
                 f"| {_pct(e['cv_guests']['bias_mean'])} | {e['cv_guests']['mase_mean']:.2f} | "
                 f"{_pct(e['holdout']['guests']['wmape'])} | {_pct(e['holdout']['nights']['wmape'])} | "
                 f"{_pct(e['holdout']['guests']['bias'])} |")
    L += ["", "![WMAPE by model](figures/wmape_models.png)", "",
          "## Country reconciliation (origin country vs guest nationality)", "",
          "| Level | Parametric CV WMAPE | Hybrid CV WMAPE | Parametric holdout | Hybrid holdout |",
          "| :- | -: | -: | -: | -: |"]
    for lvl, r in out["reconciliation"]["by_level"].items():
        L.append(f"| {LEVEL_LABELS[lvl]} | {_pct(r['parametric']['cv_wmape'])} | {_pct(r['hybrid']['cv_wmape'])} | "
                 f"{_pct(r['parametric']['holdout_wmape'])} | {_pct(r['hybrid']['holdout_wmape'])} |")
    L += ["", f"Best level for the transparent chain alone: **{LEVEL_LABELS[out['reconciliation']['best_level_for_parametric_chain']]}**.",
          "", "![Reconciliation](figures/wmape_reconciliation.png)", ""]
    ch = out["chosen"]
    L += ["## Errors by season (selected model)", "", "| Season | CV WMAPE | Holdout WMAPE | Holdout bias |",
          "| :- | -: | -: | -: |"]
    for s in settings.seasons():
        L.append(f"| {s} | {_pct(ch['cv']['guests']['by_season'][s]['wmape'])} | "
                 f"{_pct(ch['holdout']['guests']['by_season'][s]['wmape'])} | "
                 f"{_pct(ch['holdout']['guests']['by_season'][s]['bias'])} |")
    cov = ch["coverage_holdout"]
    L += ["", "![By season](figures/wmape_seasons.png)", "",
          "## Prediction intervals (split-conformal)", "",
          "Calibrated on out-of-fold residuals from the three rolling folds (per market where it has >= 30 residuals, "
          "otherwise pooled by region), then checked once on the 2025 holdout.", "",
          "| Quantity | Nominal 80% | Empirical 80% | Nominal 95% | Empirical 95% |", "| :- | -: | -: | -: | -: |"]
    for k in ("guests", "nights"):
        L.append(f"| {k} | 80% | {_pct(cov[k]['0.8'], 0)} | 95% | {_pct(cov[k]['0.95'], 0)} |")
    L += ["", "![Top 10 markets](figures/backtest_top10.png)", "",
          "## The five worst markets (2025 holdout) and why", "",
          "| Market | WMAPE | Bias | Share of guests | Explanation |", "| :- | -: | -: | -: | :- |"]
    for w in ch["worst_markets_holdout"]:
        L.append(f"| {w['name']} ({w['nationality']}) | {_pct(w['wmape'])} | {_pct(w['bias'])} | "
                 f"{_pct(w['guest_share'])} | {w['explanation']} |")
    L += ["", "![By market](figures/wmape_markets.png)", ""]
    return "\n".join(L)
