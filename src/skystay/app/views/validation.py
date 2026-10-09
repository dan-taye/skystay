from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from skystay import figures as fx
from skystay.app import ui
from skystay.backtest.report import fig_markets, fig_models, fig_reconciliation, fig_seasons
from skystay.backtest.runner import LEVEL_LABELS, MODEL_LABELS, MODELS
from skystay.reconcile import regions


def render() -> None:
    met = ui.report("metrics")
    ui.page_setup("Validation", "How well the simulator reproduces held-out hotel guests: rolling-origin "
                  "back-tests, an untouched holdout year and calibrated intervals"
                  + (" (on synthetic data with known ground truth)." if ui.synthetic() else "."))
    if not met:
        st.warning("Run `skystay backtest` first.")
        return
    sel = met["selection"]
    ho = met["chosen"]["holdout"]
    cov = met["chosen"]["coverage_holdout"]
    snaive = met["table"][sel["level"]]["seasonal_naive"]["holdout"]["guests"]["wmape"]
    ui.kpi_row([
        ("Selected model", sel["model_label"], sel["level_label"]),
        ("Holdout WMAPE, guests", ui.pct(ho["guests"]["total"]["wmape"]), f"seasonal naive: {ui.pct(snaive)}"),
        ("Holdout WMAPE, nights", ui.pct(ho["nights"]["total"]["wmape"]),
         f"bias {ho['nights']['total']['bias']:+.1%}"),
        ("80% interval coverage", ui.pct(cov["guests"]["0.8"], 0), "share of holdout months inside the 80% range"),
        ("95% interval coverage", ui.pct(cov["guests"]["0.95"], 0), "share inside the 95% range"),
    ])
    st.write("")
    df = ui.backtest_chosen()
    if not df.empty:
        top = df.groupby("nationality")["guests"].sum().sort_values(ascending=False).index.tolist()
        mk = st.selectbox("Market", top, key="val_market", format_func=lambda k: f"{regions.label(k)} ({k})")
        g = df[df["nationality"] == mk].sort_values("month")
        fig = go.Figure()
        fig.add_scatter(mode="lines", x=g["month"], y=g["guests_hi80"], line={"width": 0}, showlegend=False, hoverinfo="skip")
        fig.add_scatter(mode="lines", x=g["month"], y=g["guests_lo80"], line={"width": 0}, fill="tonexty", fillcolor=fx.BAND,
                        name="80% interval", hoverinfo="skip")
        fig.add_scatter(mode="lines", x=g["month"], y=g["pred_guests"], name="Predicted (out of sample)",
                        line={"color": fx.SERIES[0], "width": 2}, hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra></extra>")
        fig.add_scatter(mode="lines", x=g["month"], y=g["guests"], name="Actual", line={"color": fx.INK, "width": 1.6},
                        hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra></extra>")
        hstart = pd.Timestamp(next(f for f in met["folds"] if f["holdout"])["cutoff"])
        fig.add_vline(x=hstart, line={"color": fx.AXIS, "dash": "dot"})
        fig.add_annotation(x=hstart, y=1, yref="paper", text="holdout starts", showarrow=False, xanchor="left",
                           font={"color": fx.MUTED, "size": 11})
        fx.layout(fig, f"{regions.label(mk)}: actual vs back-test prediction, monthly hotel guests", height=360)
        fig.update_yaxes(title="Hotel guests per month")
        ui.chart(fig, "val_series")
    c1, c2 = st.columns(2)
    with c1:
        f = fig_models(met)
        f.update_layout(title_text=f"WMAPE by model, {LEVEL_LABELS[sel['level']]} (lower is better)")
        ui.chart(f, "val_models")
    with c2:
        f = fig_reconciliation(met)
        f.update_layout(title_text="Country reconciliation: CV WMAPE on guests")
        ui.chart(f, "val_recon")
    t = met["table"][sel["level"]]
    rows = []
    for m in MODELS:
        rows.append({"Model": MODEL_LABELS[m], "CV WMAPE guests": t[m]["cv_guests"]["wmape_mean"],
                     "CV WMAPE nights": t[m]["cv_nights"]["wmape_mean"], "CV MASE": t[m]["cv_guests"]["mase_mean"],
                     "Holdout WMAPE guests": t[m]["holdout"]["guests"]["wmape"],
                     "Holdout bias": t[m]["holdout"]["guests"]["bias"]})
    st.markdown(f"**WMAPE against every baseline** ({LEVEL_LABELS[sel['level']]}; lower is better)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="percent") for c in
        ("CV WMAPE guests", "CV WMAPE nights", "Holdout WMAPE guests", "Holdout bias")} | {
        "CV MASE": st.column_config.NumberColumn(format="%.2f")})
    c1, c2 = st.columns([1, 1.2])
    with c1:
        f = fig_seasons(met)
        f.update_layout(title_text="Selected model: WMAPE by season")
        ui.chart(f, "val_seasons")
        st.markdown("**The five worst markets in the holdout, and why**")
        st.dataframe(pd.DataFrame([{"Market": w["name"], "WMAPE": w["wmape"], "Bias": w["bias"],
                                    "Why": w["explanation"]} for w in met["chosen"]["worst_markets_holdout"]]),
                     hide_index=True, width="stretch", column_config={
                         "WMAPE": st.column_config.NumberColumn(format="percent"),
                         "Bias": st.column_config.NumberColumn(format="percent")})
    with c2:
        f = fig_markets(met)
        f.update_layout(title_text="Selected model: holdout WMAPE by market")
        ui.chart(f, "val_markets")
    rec = ui.report("recovery")
    if rec:
        st.markdown("**Parameter recovery on synthetic data with known ground truth**")
        st.dataframe(pd.DataFrame([{"Parameter": k.replace("_", " "), "What is compared": v["description"],
                                    "Statistic": v["statistic"].replace("_", " "), "Result": v["value"],
                                    "Tolerance": v["tolerance"], "Pass": "yes" if v["pass"] else "NO"}
                                   for k, v in rec["checks"].items()]), hide_index=True, width="stretch",
                     column_config={"Result": st.column_config.NumberColumn(format="%.3f")})
        st.caption(rec["note"])
    ui.synthetic_note()
