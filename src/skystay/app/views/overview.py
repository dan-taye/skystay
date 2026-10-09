from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from skystay import figures as fx
from skystay.app import ui


def render() -> None:
    e = ui.engine()
    met = ui.report("metrics")
    mod = ui.report("model")
    ui.page_setup(f"{ui.product()}: from flights to hotel guests",
                  "A scenario simulator that links Abu Dhabi air capacity to hotel demand, by source market and "
                  "month, with honest uncertainty.")
    bl = e.baseline
    t = bl.totals
    ho = met.get("chosen", {}).get("holdout", {}).get("guests", {}).get("total", {})
    ui.kpi_row([
        ("Hotel guests, reference year", ui.n0(t["guests"]),
         f"80%: {ui.n0(t['guests_interval']['lo80'])} to {ui.n0(t['guests_interval']['hi80'])}"),
        ("Guest nights, reference year", ui.n0(t["nights"]),
         f"80%: {ui.n0(t['nights_interval']['lo80'])} to {ui.n0(t['nights_interval']['hi80'])}"),
        ("Linked to flights into Abu Dhabi", ui.pct(1 - mod["last12"]["non_air_share_total"], 0),
         "the rest arrive by land, via other UAE airports, or are UAE residents"),
        ("Back-test error (WMAPE)", ui.pct(ho.get("wmape", float("nan"))),
         "monthly guests by market, untouched holdout year"),
    ])
    st.write("")
    st.markdown(
        '<div class="ss-chain"><b>The chain:</b> scheduled seats &rarr; arriving passengers (load factor) &rarr; '
        "minus transfer/transit &rarr; visitors by nationality (learned origin-to-nationality bridge) &rarr; hotel "
        "guests (capture rate) + non-air guests &rarr; guest nights (length of stay).</div>",
        unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("The problem")
        st.markdown(
            "- Most international visitors arrive by air, but only some arriving passengers become hotel guests: "
            "many are transferring, returning residents or staying with friends and family.\n"
            "- Flights are counted by **departure country**; hotel guests by **nationality**. A flight from London "
            "carries many nationalities, so the two numbers never line up directly.\n"
            "- Flights and hotels are planned by different teams with different numbers. Nobody can answer "
            "*\"what happens to hotel demand if we add this route?\"* with confidence.")
        st.subheader("What the simulator answers")
        st.markdown(
            "- How many hotel guests and nights a new route, extra frequency or bigger aircraft brings, by "
            "market and month, with an 80% and 95% range.\n"
            "- Which markets fill the most hotel rooms per seat, and which seasons are most exposed to capacity "
            "cuts.\n- Which assumptions drive the answer, each one visible and adjustable.")
    with right:
        g = bl.arrays["guests"].sum(axis=1)
        order = g.argsort()[::-1][:12][::-1]
        colors = ui.market_colors()
        names = [ui.name(e.nats[j]) for j in order]
        fig = go.Figure(go.Bar(x=g[order], y=names, orientation="h",
                               marker_color=[colors[e.nats[j]] for j in order],
                               text=[ui.n0(v) for v in g[order]], textposition="outside",
                               textfont={"color": fx.INK_2},
                               hovertemplate="%{y}: %{x:,.0f} guests<extra></extra>"))
        fx.layout(fig, "Top source markets by hotel guests (model)", height=430, legend=False)
        fig.update_xaxes(title="Hotel guests per year", showgrid=True, gridcolor=fx.GRID)
        fig.update_layout(margin={"r": 70})
        ui.chart(fig, "ov_top")
    m = bl.monthly
    fig = go.Figure()
    fig.add_scatter(x=m["month"], y=m["guests_hi80"], line={"width": 0}, showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=m["month"], y=m["guests_lo80"], line={"width": 0}, fill="tonexty", fillcolor=fx.BAND,
                    name="80% range", hoverinfo="skip")
    fig.add_scatter(x=m["month"], y=m["guests"], line={"color": fx.SERIES[0], "width": 2}, name="Hotel guests",
                    hovertemplate="%{x|%b %Y}: %{y:,.0f} guests<extra></extra>")
    fx.layout(fig, "Hotel guests by month, reference year (model, with 80% range)", height=320)
    fig.update_yaxes(title="Hotel guests per month", rangemode="tozero")
    fig.update_xaxes(title=None, dtick="M1", tickformat="%b")
    ui.chart(fig, "ov_month")
    ui.synthetic_note()
