"""Shared UI helpers: cached resources, styling, badge, KPI cards, formatting."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from skystay import figures as fx
from skystay import paths, settings
from skystay.reconcile import regions

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
                 "toImageButtonOptions": {"format": "png", "scale": 2}}


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading the model...")
def engine():
    from skystay import train
    from skystay.scenario.engine import Engine

    return Engine(train.load())


@st.cache_data(show_spinner=False)
def report(name: str) -> dict:
    f = paths.REPORTS / f"{name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


@st.cache_data(show_spinner=False)
def assumptions() -> dict:
    return yaml.safe_load(paths.ASSUMPTIONS_FILE.read_text(encoding="utf-8")) if paths.ASSUMPTIONS_FILE.exists() else {}


@st.cache_data(show_spinner=False)
def doc(name: str) -> str:
    f = paths.DOCS / name
    return f.read_text(encoding="utf-8") if f.exists() else ""


@st.cache_data(show_spinner=False)
def backtest_chosen() -> pd.DataFrame:
    f = paths.PROCESSED / "backtest_chosen.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


def product() -> str:
    return settings.product_name()


def synthetic() -> bool:
    return settings.is_synthetic()


# ------------------------------------------------------------------ styling
CSS = """
<style>
#MainMenu, footer, [data-testid="stStatusWidget"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"] {visibility: hidden;}
.block-container {padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1500px;}
h1 {font-size: 1.9rem !important; font-weight: 650 !important; letter-spacing: -0.01em;}
h2 {font-size: 1.25rem !important; font-weight: 600 !important;}
h3 {font-size: 1.05rem !important; font-weight: 600 !important;}
.ss-sub {color: #52514e; font-size: 1.0rem; margin-top: -0.6rem; margin-bottom: 1rem;}
.ss-badge {position: fixed; top: 0.75rem; right: 1.2rem; z-index: 999999; background: #fff3d6;
  color: #7a4b00; border: 1px solid #eda100; border-radius: 6px; padding: 4px 10px; font-weight: 700;
  font-size: 0.78rem; letter-spacing: 0.04em; box-shadow: 0 1px 2px rgba(0,0,0,0.06);}
.ss-kpi {background: #ffffff; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 14px 16px;
  height: 100%;}
.ss-kpi .lab {color: #52514e; font-size: 0.82rem; margin-bottom: 4px;}
.ss-kpi .val {color: #0b0b0b; font-size: 1.55rem; font-weight: 650; line-height: 1.2;}
.ss-kpi .sub {color: #898781; font-size: 0.78rem; margin-top: 4px;}
.ss-kpi .up {color: #006300;} .ss-kpi .down {color: #b42318;}
.ss-note {color: #52514e; font-size: 0.86rem;}
.ss-chain {background: #f3f2ee; border-radius: 8px; padding: 10px 14px; color: #0b0b0b; font-size: 0.95rem;}
.ss-pill {display: inline-block; background: #e8f0fb; color: #184f95; border-radius: 999px; padding: 2px 10px;
  font-size: 0.78rem; margin-right: 6px;}
</style>
"""


def page_setup(title: str, subtitle: str | None = None) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if synthetic():
        st.markdown('<div class="ss-badge" data-testid="synthetic-badge">SYNTHETIC DEMO DATA</div>',
                    unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f'<div class="ss-sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    cls = {"up": "up", "down": "down"}.get(tone, "")
    return (f'<div class="ss-kpi"><div class="lab">{label}</div><div class="val {cls}">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for c, it in zip(cols, items, strict=True):
        c.markdown(kpi(*it), unsafe_allow_html=True)


def chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)


# ------------------------------------------------------------------ formatting
def n0(x: float) -> str:
    return f"{x:,.0f}"


def signed(x: float) -> str:
    return f"{x:+,.0f}"


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def name(key: str) -> str:
    return regions.label(key)


def market_colors() -> dict[str, str]:
    e = engine()
    g = dict(zip(e.nats, e.baseline.arrays["guests"].sum(axis=1), strict=True))
    return fx.market_colors(fx.top_markets(g), e.nats)


def synthetic_note() -> None:
    if synthetic():
        st.caption("All numbers are produced by the pipeline from SYNTHETIC DEMO DATA with known ground truth. "
                   "They demonstrate the method and say nothing about real Abu Dhabi demand.")
