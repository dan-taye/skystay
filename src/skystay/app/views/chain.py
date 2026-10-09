from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from skystay import figures as fx
from skystay import settings
from skystay.app import ui
from skystay.reconcile import regions


def _fmt_iv(iv) -> str:
    if not iv or iv[0] is None:
        return "-"
    return f"{iv[0]:,.3g} to {iv[1]:,.3g}"


def render() -> None:
    e = ui.engine()
    ui.page_setup("Conversion chain", "Follow one market's seats all the way to hotel guest nights. Every step's "
                  "assumption is listed below with its source and interval.")
    a = e.baseline.arrays
    origins = [o for i, o in enumerate(a["origins"]) if a["S"][i].sum() > 0]
    default = int(np.argmax([a["flow"][a["origins"].index(o)].sum() for o in origins]))
    c1, c2, _ = st.columns([1, 1, 2])
    o = c1.selectbox("Origin market (departure country)", origins, index=default, key="ch_origin",
                     format_func=lambda k: f"{regions.label(k)} ({k})")
    season = c2.selectbox("Season", ["Whole year"] + list(settings.seasons()), key="ch_season")
    mask = np.ones(12, dtype=bool) if season == "Whole year" else e.season_mask(season)
    i = a["origins"].index(o)
    S = a["S"][i, mask].sum()
    P = a["P"][i, mask].sum()
    T = a["T"][i, mask].sum()
    flow = a["flow"][i][:, mask] * a["factor"][:, mask]  # hotel guests by nationality from this origin
    by_nat = flow.sum(axis=1)
    nights = (flow * a["alos"][:, mask]).sum(axis=1)
    order = np.argsort(-by_nat)
    top = [j for j in order[:3] if by_nat[j] > 0.5]
    rest = [j for j in order if j not in top and by_nat[j] > 0]
    guests = by_nat.sum()
    not_hotel = max(T - guests, 0.0)

    labels = [f"Seats<br>{S:,.0f}", f"Passengers<br>{P:,.0f}", f"Empty seats<br>{S - P:,.0f}",
              f"Transfer / transit<br>{P - T:,.0f}", f"Visitors and residents<br>entering Abu Dhabi<br>{T:,.0f}",
              f"Not in hotels<br>(residents, friends & family,<br>other lodging) {not_hotel:,.0f}"]
    src, dst, val, col = [0, 0, 1, 1, 4], [1, 2, 3, 4, 5], [P, S - P, P - T, T, not_hotel], [
        "rgba(42,120,214,0.35)", "rgba(195,194,183,0.45)", "rgba(195,194,183,0.45)", "rgba(42,120,214,0.35)",
        "rgba(195,194,183,0.45)"]
    node_colors = [fx.SERIES[0], fx.SERIES[0], fx.OTHER, fx.OTHER, fx.SERIES[0], fx.OTHER]
    nights_idx = None
    groups = [([j], regions.label(e.nats[j])) for j in top] + ([(rest, "Other nationalities")] if rest else [])
    for js, lab in groups:
        g = float(by_nat[js].sum())
        labels.append(f"Hotel guests: {lab}<br>{g:,.0f}")
        node_colors.append(fx.SERIES[2])
        k = len(labels) - 1
        src.append(4)
        dst.append(k)
        val.append(g)
        col.append("rgba(27,175,122,0.40)")
    total_nights = float(nights.sum())
    labels.append(f"Guest nights<br>{total_nights:,.0f}<br>(x length of stay)")
    node_colors.append(fx.SERIES[6])
    nights_idx = len(labels) - 1
    for gi, (js, _) in enumerate(groups):
        src.append(6 + gi)
        dst.append(nights_idx)
        val.append(float(nights[js].sum()))
        col.append("rgba(74,58,167,0.30)")
    ng = len(groups)
    xs = [0.01, 0.24, 0.50, 0.50, 0.50, 0.76] + [0.76] * ng + [0.99]
    ys = [0.45, 0.40, 0.93, 0.72, 0.25, 0.80] + [0.08 + 0.14 * k for k in range(ng)] + [0.30]
    fig = go.Figure(go.Sankey(
        arrangement="fixed", valueformat=",.0f",
        node={"label": labels, "color": node_colors, "pad": 22, "thickness": 16, "x": xs, "y": ys,
              "line": {"color": "rgba(0,0,0,0)", "width": 0}},
        link={"source": src, "target": dst, "value": val, "color": col},
    ))
    fx.layout(fig, f"{regions.label(o)}: seats to guest nights ({season.lower()}, reference year)", height=560,
              legend=False)
    fig.update_layout(font={"size": 13, "color": fx.INK})
    ui.chart(fig, "ch_sankey")
    st.caption("Hotel guests from this origin include other nationalities carried by the route (the learned "
               "origin-to-nationality bridge). Guest nights are guests multiplied by each nationality's length of "
               "stay, so that node is measured in nights, not people.")

    asm = ui.assumptions()
    lf_a = asm.get("load_factor", {}).get(o, {})
    tr_a = asm.get("transfer_share", {}).get(o, {})
    nat = o if o in e.nats else "OTHER"
    al_a = asm.get("length_of_stay", {}).get(nat, {})
    cp_a = asm.get("capture_rate", {}).get(nat, {})
    rows = [
        ("Seats to passengers", "Load factor", f"{P / S:.1%}" if S else "-", _fmt_iv(lf_a.get("interval_80")),
         lf_a.get("method", "-"), lf_a.get("n_obs", "-")),
        ("Passengers to visitors", "Transfer / transit share", f"{(P - T) / P:.1%}" if P else "-",
         _fmt_iv(tr_a.get("interval_80")), tr_a.get("method", "-"), tr_a.get("n_obs", "-")),
        ("Visitors to hotel guests", "Hotel guests per terminating passenger (all nationalities)",
         f"{guests / T:.3f}" if T else "-", "see capture rate", "learned bridge x capture, x ML correction",
         "-"),
        ("", f"Capture rate, {regions.label(nat)} nationals", f"{cp_a.get('value', '-')}",
         _fmt_iv(cp_a.get("interval_80")), cp_a.get("method", "-"), cp_a.get("n_obs", "-")),
        ("Guests to nights", f"Length of stay, {regions.label(nat)} nationals",
         f"{al_a.get('value', '-')} nights", _fmt_iv(al_a.get("interval_80")), al_a.get("method", "-"),
         al_a.get("n_obs", "-")),
        ("Not moved by flights", f"Non-air guests, {regions.label(nat)} nationals",
         f"{asm.get('non_air_baseline', {}).get(nat, {}).get('value', '-')}/month",
         _fmt_iv(asm.get("non_air_baseline", {}).get(nat, {}).get("interval_80")),
         "estimated from data; levers do not move it", "-"),
    ]
    st.subheader("Assumptions at each step")
    st.dataframe(pd.DataFrame(rows, columns=["Step", "Assumption", "Value", "80% interval", "Method", "Months of data"]),
                 hide_index=True, width="stretch")
    st.caption("Source: " + ("synthetic stand-in data (SYNTHETIC DEMO DATA)" if ui.synthetic() else "real data")
               + ". Full ledger: Data & assumptions page and config/assumptions.yaml.")
