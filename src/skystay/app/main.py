"""Streamlit entry point: `skystay app` (or `streamlit run src/skystay/app/main.py`)."""

from __future__ import annotations

import streamlit as st

from skystay import settings
from skystay.app.views import assumptions, brief, chain, how, overview, sensitivity, simulator, validation

st.set_page_config(page_title=f"{settings.product_name()} | flight-to-hotel simulator", page_icon=":material/hotel:",
                   layout="wide", initial_sidebar_state="collapsed")

PAGES = [
    st.Page(overview.render, title="Overview", icon=":material/dashboard:", default=True),
    st.Page(how.render, title="How it works", icon=":material/account_tree:", url_path="how-it-works"),
    st.Page(simulator.render, title="Scenario simulator", icon=":material/tune:", url_path="simulator"),
    st.Page(chain.render, title="Conversion chain", icon=":material/conversion_path:", url_path="chain"),
    st.Page(sensitivity.render, title="Sensitivity", icon=":material/insights:", url_path="sensitivity"),
    st.Page(validation.render, title="Validation", icon=":material/fact_check:", url_path="validation"),
    st.Page(assumptions.render, title="Data & assumptions", icon=":material/dataset:", url_path="data"),
    st.Page(brief.render, title="Planner brief", icon=":material/description:", url_path="brief"),
]

st.navigation(PAGES, position="top").run()
