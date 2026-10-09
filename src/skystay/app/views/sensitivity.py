from __future__ import annotations

import streamlit as st

from skystay.app import ui
from skystay.sensitivity import run as sens


def render() -> None:
    ui.page_setup("Sensitivity", "Which levers move hotel demand most, locally (one at a time) and globally "
                  "(all at once, Sobol).")
    s = ui.report("sensitivity")
    if not s:
        st.warning("Run `skystay sensitivity` to produce reports/sensitivity.json.")
        return
    sob = s["sobol"]["factors"]
    st.markdown(
        f'<div class="ss-chain"><b>What matters most:</b> {sob[0]["label"].lower()} explains about '
        f'{sob[0]["ST"]:.0%} of the variation in annual guest nights across plausible lever ranges, then '
        f'{sob[1]["label"].lower()} ({sob[1]["ST"]:.0%}) and {sob[2]["label"].lower()} ({sob[2]["ST"]:.0%}). '
        "Calendar and seasonal shifts matter far less for the annual total.</div>", unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        which = st.segmented_control("Tornado step", ["pct10", "sd1"], default="pct10", key="sens_step",
                                     format_func=lambda x: "+/-10%" if x == "pct10" else "+/-1 historical SD")
        ui.chart(sens.fig_tornado(s["tornado"], which or "pct10"), "sens_tornado")
        st.caption(s["tornado"]["note"])
    with c2:
        ui.chart(sens.fig_sobol(s["sobol"]), "sens_sobol")
        st.caption(f"{s['sobol']['method']}; N = {s['sobol']['N']:,} ({s['sobol']['evaluations']:,} model runs). "
                   "Ranges: " + "; ".join(f"{f['label']} {f['range'][0]:g} to {f['range'][1]:g} {f['unit']}"
                                          for f in sob))
    c1, c2 = st.columns([1.15, 1])
    with c1:
        ui.chart(sens.fig_per1000(s["per_1000_seats"]), "sens_per1000")
        st.caption(s["per_1000_seats"]["method"])
    with c2:
        ui.chart(sens.fig_vulnerability(s["vulnerability"]), "sens_vuln")
        st.caption(s["vulnerability"]["method"])
        rows = [{"Season": r["season"], "Guest nights lost": f"{-r['delta_nights']:,.0f}",
                 "Share of season": f"{-r['pct_of_season']:.1%}",
                 "Most affected": ", ".join(m["name"] for m in r["most_affected"])} for r in s["vulnerability"]["rows"]]
        st.dataframe(rows, hide_index=True, width="stretch")
    ui.synthetic_note()
