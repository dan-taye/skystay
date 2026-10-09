from __future__ import annotations

import streamlit as st

from skystay.app import ui

USE_PLAN = [
    ("Who uses it", "DCT tourism planning and route-development teams, with Abu Dhabi Airports and partner "
                    "airlines working from the same numbers."),
    ("Decisions it changes", "Which routes to incentivise, how much capacity to protect in each season, where to "
                             "aim source-market campaigns, and what hotel demand to plan staffing and inventory for."),
    ("How it runs", "Monthly refresh: new schedule, traffic and hotel files go into data/raw, one command re-runs "
                    "the pipeline, back-test and app. Every assumption stays visible and adjustable."),
    ("Next steps", "Load real data, re-run the back-test on real history, review the "
                            "assumption ledger with DCT mentors, then pilot it on one planning cycle."),
]


def render() -> None:
    ui.page_setup("Planner brief", "Plain-language findings, generated from the pipeline's reports by a fixed "
                  "template (no language model).")
    s = ui.report("sensitivity")
    if not s:
        st.warning("Run `skystay sensitivity` first.")
        return
    sec = s["planner_brief"]["sections"]
    st.markdown("### Real-life use plan")
    cols = st.columns(len(USE_PLAN))
    for col, (title, text) in zip(cols, USE_PLAN, strict=True):
        col.markdown(f'<div class="ss-kpi"><div style="font-weight:650;font-size:1.02rem;color:#184f95;'
                     f'margin-bottom:6px">{title}</div><div style="color:#0b0b0b;font-size:0.95rem;line-height:1.45">'
                     f"{text}</div></div>", unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    for col, keys in ((c1, ("best_routes", "new_markets", "seasonal_risk")), (c2, ("what_matters", "confidence"))):
        with col:
            for k in keys:
                st.markdown(f"### {sec[k]['title']}")
                st.markdown(sec[k]["text"])
    st.markdown(f"### {sec['do_differently']['title']}")
    st.markdown("\n".join(f"{i}. {b}" for i, b in enumerate(sec["do_differently"]["bullets"], 1)))
    st.write("")
    st.download_button("Download this brief (Markdown)", (ui.doc("../reports/planner_brief.md")).encode(),
                       file_name=f"{ui.product()}_planner_brief.md", mime="text/markdown", key="brief_dl")
    ui.synthetic_note()
