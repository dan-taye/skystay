from __future__ import annotations

import streamlit as st

from skystay.app import ui


def _box(x, y, w, h, title, lines, accent=False):
    stroke = "#2a78d6" if accent else "#c3c2b7"
    fill = "#eef4fc" if accent else "#ffffff"
    t = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
         f'<text x="{x + 14}" y="{y + 28}" font-size="17" font-weight="650" fill="#0b0b0b">{title}</text>']
    for i, ln in enumerate(lines):
        t.append(f'<text x="{x + 14}" y="{y + 52 + i * 20}" font-size="13.5" fill="#52514e">{ln}</text>')
    return "".join(t)


def _arrow(x1, y1, x2, y2, dashed=False):
    d = ' stroke-dasharray="5 4"' if dashed else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#52514e" stroke-width="1.6"{d} '
            f'marker-end="url(#ah)"/>')


def diagram(level_label: str, model_label: str, synthetic: bool) -> str:
    src = "Synthetic stand-in" if synthetic else "Real data"
    parts = [
        '<svg viewBox="0 0 1260 470" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="System design: data to canonical store to parametric and ML layers to scenario engine to app" '
        'style="font-family: system-ui, -apple-system, Segoe UI, sans-serif">',
        '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker></defs>',
        _box(10, 40, 200, 170, src, ["Flight schedule (seats)", "Passenger traffic", "Hotel guests & nights",
                                    "Events, holidays,", "seasonality profile"]),
        _box(250, 40, 210, 170, "Canonical store", ["DuckDB tables", "Schema-mapped adapters", "Data-quality checks",
                                                   "Country keys & regions"]),
        _box(500, 10, 250, 140, "Parametric chain", ["Load factor, transfer share", "Origin to nationality bridge",
                                                    f"({level_label})", "Capture, non-air base, ALOS"], accent=True),
        _box(500, 170, 250, 110, "ML layer (LightGBM)", ["Seasonality, holidays, events", "Monotone in seats",
                                                         "SHAP explanations"]),
        _box(790, 70, 200, 140, "Hybrid model", [f"{model_label}", "chain x ML correction", "chosen by back-test"],
             accent=True),
        _box(1030, 40, 220, 170, "Scenario engine", ["11 levers", "Monte Carlo (500 draws)", "Conformal 80/95%",
                                                    "Guardrails, cold start"], accent=True),
        _box(500, 330, 250, 120, "Back-test & validation", ["Rolling-origin folds", "WMAPE vs 4 baselines",
                                                            "Interval calibration"]),
        _box(790, 330, 200, 120, "Sensitivity", ["Tornado", "Sobol indices", "Guests per 1,000 seats"]),
        _box(1030, 300, 220, 150, "Simulator app", ["What-if by market & month", "A vs B comparison",
                                                   "Planner brief", "CSV / JSON export"], accent=True),
        _arrow(210, 125, 248, 125), _arrow(460, 95, 498, 80), _arrow(460, 150, 498, 220),
        _arrow(750, 80, 788, 120), _arrow(750, 225, 788, 170), _arrow(990, 140, 1028, 125),
        _arrow(1140, 210, 1140, 298), _arrow(990, 390, 1028, 380), _arrow(1060, 210, 920, 328, dashed=True),
        _arrow(625, 330, 625, 282, dashed=True), _arrow(700, 330, 840, 212, dashed=True),
        '<text x="760" y="300" font-size="12" fill="#898781">selects level &amp; model</text>',
        "</svg>",
    ]
    return "".join(parts)


def render() -> None:
    ui.page_setup("How it works", "From raw aggregates to a what-if answer, with every step visible.")
    met = ui.report("metrics")
    sel = met.get("selection", {})
    st.markdown(diagram(sel.get("level_label", "L2 learned bridge"), sel.get("model_label", "Hybrid"), ui.synthetic()),
                unsafe_allow_html=True)
    st.write("")
    st.subheader("How to use it in three steps")
    c1, c2, c3 = st.columns(3)
    c1.markdown(ui.kpi("Step 1", "Pull a lever", "On the Scenario simulator: add a route, change frequency, aircraft, "
                       "load factor, capture, length of stay, seasons or events."), unsafe_allow_html=True)
    c2.markdown(ui.kpi("Step 2", "Read the change", "Hotel guests and nights update by market and month, with 80% "
                       "and 95% ranges and warnings when you leave the data's range."), unsafe_allow_html=True)
    c3.markdown(ui.kpi("Step 3", "Check and share", "See what drives it (Sensitivity, Conversion chain), compare "
                       "Scenario A vs B, export CSV/JSON, read the Planner brief."), unsafe_allow_html=True)
    st.write("")
    st.subheader("Why two layers")
    st.markdown(
        "The **parametric chain** is the auditable part: every number in it (load factor, transfer share, capture "
        "rate, length of stay, non-air baseline) is estimated from data, shown with an interval in the assumption "
        "ledger, and adjustable. The **ML layer** learns what the chain misses (holiday timing, events, market "
        "quirks) as a correction factor that is constrained never to reduce guests when seats rise. The back-test "
        "picks the combination that best reproduces held-out hotel guests.")
    ui.synthetic_note()
