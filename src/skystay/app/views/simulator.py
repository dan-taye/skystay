from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from skystay import figures as fx
from skystay import settings
from skystay.app import ui
from skystay.reconcile import regions
from skystay.scenario import presets as preset_mod
from skystay.scenario.engine import ScenarioError
from skystay.scenario.levers import (
    Aircraft,
    Alos,
    Capture,
    DiscontinueRoute,
    Event,
    Frequency,
    LoadFactor,
    MixShift,
    NewRoute,
    Scenario,
    SeasonShift,
    TransferShare,
)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NONE = "(none)"


def seasons() -> list[str]:
    return list(settings.seasons().keys())


def top_origin(e) -> str:
    """Origin whose flights generate the most hotel guests in the reference year."""
    a = e.baseline.arrays
    return a["origins"][int(np.argmax(a["flow"].sum(axis=(1, 2))))]


def defaults() -> dict:
    s = seasons()
    return {
        "nr_on": False, "nr_origin": "CA", "nr_aircraft": "B787-9", "nr_freq": 2, "nr_start": "Jan",
        "dr_routes": [], "dr_from": "Jan",
        "cap_market": "ALL", "cap_pct": 0, "cap_season": "ALL",
        "ac_market": NONE, "ac_type": "B787-9",
        "lf_market": "ALL", "lf_pp": 0.0, "tr_market": "ALL", "tr_pp": 0.0,
        "capt_market": "ALL", "capt_pct": 0, "alos_market": "ALL", "alos_pct": 0,
        "ss_market": "ALL", "ss_from": s[2], "ss_to": s[0], "ss_pct": 0,
        "mix_from": NONE, "mix_to": NONE, "mix_pct": 0,
        "ev_month": "Jun", "ev_days": 0,
    }


def preset_values(name: str, e) -> dict:
    s = seasons()
    if name == "new_route":
        return {"nr_on": True, "nr_origin": "CA", "nr_aircraft": "B787-9", "nr_freq": 2, "nr_start": "Jan"}
    if name == "winter_cut":
        return {"cap_market": top_origin(e), "cap_pct": -30, "cap_season": s[0]}
    return {}


def init_state(e) -> None:
    preset = st.query_params.get("preset", "")
    if st.session_state.get("_sim_init") == preset:
        return
    for k, v in defaults().items():
        st.session_state[k] = v
    for k, v in preset_values(preset, e).items():
        st.session_state[k] = v
    st.session_state["_sim_init"] = preset


def build_levers(e) -> list:
    s = st.session_state
    L = []
    if s["nr_on"]:
        L.append(NewRoute(origin=s["nr_origin"], aircraft=s["nr_aircraft"], weekly_freq=float(s["nr_freq"]),
                          start_month=MONTHS.index(s["nr_start"]) + 1))
    for rid in s["dr_routes"]:
        L.append(DiscontinueRoute(route_id=rid, from_month=MONTHS.index(s["dr_from"]) + 1))
    if s["cap_pct"] != 0:
        L.append(Frequency(market=s["cap_market"], pct=float(s["cap_pct"]), season=s["cap_season"]))
    if s["ac_market"] != NONE:
        L.append(Aircraft(market=s["ac_market"], aircraft=s["ac_type"]))
    if s["lf_pp"] != 0:
        L.append(LoadFactor(market=s["lf_market"], delta_pp=float(s["lf_pp"])))
    if s["tr_pp"] != 0:
        L.append(TransferShare(market=s["tr_market"], delta_pp=float(s["tr_pp"])))
    if s["capt_pct"] != 0:
        L.append(Capture(market=s["capt_market"], pct=float(s["capt_pct"])))
    if s["alos_pct"] != 0:
        L.append(Alos(market=s["alos_market"], pct=float(s["alos_pct"])))
    if s["ss_pct"] != 0 and s["ss_from"] != s["ss_to"]:
        L.append(SeasonShift(market=s["ss_market"], from_season=s["ss_from"], to_season=s["ss_to"],
                             pct=float(s["ss_pct"])))
    if s["mix_pct"] != 0 and NONE not in (s["mix_from"], s["mix_to"]) and s["mix_from"] != s["mix_to"]:
        L.append(MixShift(from_market=s["mix_from"], to_market=s["mix_to"], pct=float(s["mix_pct"])))
    if s["ev_days"] != 0:
        L.append(Event(month=MONTHS.index(s["ev_month"]) + 1, days=int(s["ev_days"])))
    return L


def describe(levers: list) -> str:
    if not levers:
        return "Baseline (no change)"
    parts = []
    for lv in levers:
        k = lv.kind
        if k == "new_route":
            parts.append(f"new {lv.weekly_freq:g}x-weekly {lv.aircraft} route from {regions.label(lv.origin)}")
        elif k == "frequency":
            where = "all markets" if lv.market == "ALL" else regions.label(lv.market)
            when = "" if lv.season == "ALL" else f" in {lv.season.split(' (')[0].lower()}"
            parts.append(f"{lv.pct:+.0f}% frequency, {where}{when}")
        else:
            parts.append(k.replace("_", " "))
    return "; ".join(parts)


@st.cache_data(max_entries=64, show_spinner=False)
def run_cached(scenario_json: str):
    return ui.engine().run(Scenario.model_validate_json(scenario_json))


def _origin_options(e, include_all=True) -> list[str]:
    served = sorted({o for o in e.routes["origin"].unique()}, key=regions.label)
    return (["ALL"] if include_all else []) + served


def _fmt(k: str) -> str:
    return "All markets" if k == "ALL" else (NONE if k == NONE else f"{regions.label(k)} ({k})")


def lever_panel(e) -> None:
    origins = _origin_options(e)
    served = set(e.routes["origin"].unique())
    every = sorted([k for k in regions.region_of() if k not in ("AE", "OTHER")], key=regions.label)
    nats = ["ALL"] + sorted([n for n in e.nats], key=regions.label)
    aircraft = sorted(k for k in e.spf if k in ("A320", "A321", "B787-9", "A350-1000", "B777-300ER")) or sorted(e.spf)
    s = st.session_state
    st.markdown("#### Levers")
    with st.expander("Routes", expanded=True):
        st.toggle("Add a new route", key="nr_on", help="Default: off. Adds a route from any country, including "
                  "one with no service today (cold-start estimate).")
        c1, c2 = st.columns(2)
        c1.selectbox("Origin country", every, key="nr_origin",
                     format_func=lambda k: f"{regions.label(k)}{'' if k in served else ' (no service today)'}",
                     help="Departure country of the new route.")
        c2.selectbox("Aircraft", aircraft, key="nr_aircraft",
                     help="Sets seats per flight (typical seat count observed for this type).")
        c1.slider("Weekly flights", 1, 14, key="nr_freq", help="Default 2. Range 1 to 14 flights per week.")
        c2.selectbox("Start month", MONTHS, key="nr_start", help="First month of operation in the plan year.")
        rt = e.route_table()
        st.multiselect("Discontinue routes", rt["route_id"].tolist(), key="dr_routes",
                       format_func=lambda r: f"{r} ({regions.label(rt.set_index('route_id').loc[r, 'origin'])})",
                       help="Stop existing routes from the month below. Default: none.")
        st.selectbox("Stop from", MONTHS, key="dr_from")
    with st.expander("Capacity", expanded=s["cap_pct"] != 0 or s["ac_market"] != NONE):
        st.selectbox("Market", origins, key="cap_market", format_func=_fmt)
        st.slider("Weekly frequency change (%)", -50, 100, step=5, key="cap_pct",
                  help="Default 0%. Range -50% to +100% of weekly flights on the market's routes.")
        st.selectbox("Season", ["ALL"] + seasons(), key="cap_season",
                     format_func=lambda x: "Whole year" if x == "ALL" else x)
        st.selectbox("Swap aircraft for market", [NONE] + origins[1:], key="ac_market", format_func=_fmt,
                     help="Default: none. Every route of the market flies this aircraft type.")
        st.selectbox("New aircraft type", aircraft, key="ac_type")
    with st.expander("Conversion", expanded=any(s[k] != 0 for k in ("lf_pp", "tr_pp", "capt_pct", "alos_pct"))):
        c1, c2 = st.columns(2)
        c1.selectbox("Load factor: market", origins, key="lf_market", format_func=_fmt)
        c2.slider("Load factor (pp)", -20.0, 15.0, step=1.0, key="lf_pp",
                  help="Default 0 percentage points. Cannot push any month above 100%.")
        c1.selectbox("Transfer share: market", origins, key="tr_market", format_func=_fmt)
        c2.slider("Transfer share (pp)", -20.0, 20.0, step=1.0, key="tr_pp",
                  help="Default 0 pp. Passengers who connect onward never enter Abu Dhabi.")
        c1.selectbox("Capture rate: nationality", nats, key="capt_market", format_func=_fmt)
        c2.slider("Capture rate (%)", -30, 30, step=5, key="capt_pct",
                  help="Default 0%. Share of visitors who stay in a hotel. Beyond +/-20% is flagged as extrapolation.")
        c1.selectbox("Length of stay: nationality", nats, key="alos_market", format_func=_fmt)
        c2.slider("Length of stay (%)", -30, 30, step=5, key="alos_pct", help="Default 0%. Nights per guest.")
    with st.expander("Season & events", expanded=any(s[k] != 0 for k in ("ss_pct", "mix_pct", "ev_days"))):
        st.selectbox("Seasonal shift: market", origins, key="ss_market", format_func=_fmt)
        c1, c2 = st.columns(2)
        c1.selectbox("Move seats from", seasons(), key="ss_from")
        c2.selectbox("to", seasons(), key="ss_to")
        st.slider("Share of seats moved (%)", 0, 30, step=5, key="ss_pct", help="Default 0%.")
        c1, c2 = st.columns(2)
        c1.selectbox("Market mix: from", [NONE] + origins[1:], key="mix_from", format_func=_fmt)
        c2.selectbox("to", [NONE] + every, key="mix_to", format_func=_fmt)
        st.slider("Seats moved between markets (%)", 0, 50, step=5, key="mix_pct", help="Default 0%.")
        c1, c2 = st.columns(2)
        c1.selectbox("Event month", MONTHS, key="ev_month")
        c2.slider("Event days added", -5, 10, key="ev_days", help="Default 0. Abu Dhabi event days in that month.")
    if st.button("Reset all levers", key="reset", type="secondary"):
        for k, v in defaults().items():
            st.session_state[k] = v
        st.rerun()


def monthly_delta(res, bl, metric: str):
    key = "g_d" if metric == "guests" else "n_d"
    d = res.arrays[key].sum(axis=1) - bl.arrays[key].sum(axis=1)  # (D, 12)
    point = res.arrays[metric].sum(axis=0) - bl.arrays[metric].sum(axis=0)
    return point, np.percentile(d, 10, axis=0), np.percentile(d, 90, axis=0)


def results_panel(e, sc: Scenario, res) -> None:
    bl = e.baseline
    t = res.totals
    dg, dn = t["delta_guests_interval"], t["delta_nights_interval"]
    st.markdown(f"#### {describe(list(sc.levers))}")
    ui.kpi_row([
        ("Change in hotel guests", ui.signed(t["delta_guests"]), f"80%: {ui.signed(dg['lo80'])} to {ui.signed(dg['hi80'])}",
         "up" if t["delta_guests"] > 0 else ("down" if t["delta_guests"] < 0 else "")),
        ("Change in guest nights", ui.signed(t["delta_nights"]), f"80%: {ui.signed(dn['lo80'])} to {ui.signed(dn['hi80'])}",
         "up" if t["delta_nights"] > 0 else ("down" if t["delta_nights"] < 0 else "")),
        ("Hotel guests, plan year", ui.n0(t["guests"]),
         f"80%: {ui.n0(t['guests_interval']['lo80'])} to {ui.n0(t['guests_interval']['hi80'])}"),
        ("Seats into Abu Dhabi", ui.signed(t["seats"] - t["baseline_seats"]), f"of {ui.n0(t['baseline_seats'])} today"),
    ])
    for w in res.warnings:
        (st.info if w.startswith("cold-start") else st.warning)(w, icon=":material/info:" if w.startswith("cold")
                                                               else ":material/warning:")
    tabs = st.tabs(["By month", "By market", "Scenario A vs B", "Export"])
    with tabs[0]:
        metric = st.segmented_control("Show", ["guests", "nights"], key="sim_metric", default="guests",
                                      format_func=lambda x: "Hotel guests" if x == "guests" else "Guest nights")
        metric = metric or "guests"
        unit = "Hotel guests" if metric == "guests" else "Guest nights"
        point, lo, hi = monthly_delta(res, bl, metric)
        months = res.monthly["month"]
        fig = go.Figure(go.Bar(
            x=months, y=point, marker_color=[fx.DIV_POS if v >= 0 else fx.DIV_NEG for v in point],
            error_y={"type": "data", "symmetric": False, "array": hi - point, "arrayminus": point - lo,
                     "color": fx.INK_2, "thickness": 1},
            hovertemplate="%{x|%b}: %{y:+,.0f}<extra></extra>", name="Change"))
        fx.layout(fig, f"Change in {unit.lower()} vs baseline by month (80% range)", height=330, legend=False)
        fig.update_yaxes(title=f"{unit} vs baseline", tickformat="+,.0f", zeroline=True, zerolinecolor=fx.AXIS)
        fig.update_xaxes(dtick="M1", tickformat="%b")
        ui.chart(fig, "sim_delta_month")
        m = res.monthly
        col = metric
        fig = go.Figure()
        fig.add_scatter(mode="lines", x=m["month"], y=m[f"{col}_hi95"], line={"width": 0}, showlegend=False, hoverinfo="skip")
        fig.add_scatter(mode="lines", x=m["month"], y=m[f"{col}_lo95"], line={"width": 0}, fill="tonexty", fillcolor=fx.BAND_95,
                        name="95% range", hoverinfo="skip")
        fig.add_scatter(mode="lines", x=m["month"], y=m[f"{col}_hi80"], line={"width": 0}, showlegend=False, hoverinfo="skip")
        fig.add_scatter(mode="lines", x=m["month"], y=m[f"{col}_lo80"], line={"width": 0}, fill="tonexty", fillcolor=fx.BAND,
                        name="80% range", hoverinfo="skip")
        fig.add_scatter(mode="lines", x=m["month"], y=m[f"baseline_{col}"], name="Baseline",
                        line={"color": fx.MUTED, "width": 2, "dash": "dot"},
                        hovertemplate="%{x|%b}: %{y:,.0f}<extra>Baseline</extra>")
        fig.add_scatter(mode="lines", x=m["month"], y=m[col], name="Scenario", line={"color": fx.SERIES[0], "width": 2.5},
                        hovertemplate="%{x|%b}: %{y:,.0f}<extra>Scenario</extra>")
        fx.layout(fig, f"{unit} by month: scenario vs baseline", height=330)
        fig.update_yaxes(title=f"{unit} per month", rangemode="tozero")
        fig.update_xaxes(dtick="M1", tickformat="%b")
        ui.chart(fig, "sim_level_month")
    with tabs[1]:
        by = res.table.groupby("market").agg(guests=("guests", "sum"), delta=("delta_vs_baseline", "sum"),
                                             nights=("nights", "sum"), dn=("delta_nights", "sum")).reset_index()
        moved = by[by["delta"].abs() > 0.5]
        if moved.empty:
            moved = by.sort_values("guests", ascending=False).head(12).assign(delta=0.0)
        moved = moved.reindex(moved["delta"].abs().sort_values(ascending=False).index).head(12)
        moved = moved.iloc[::-1]
        fig = go.Figure(go.Bar(
            x=moved["delta"], y=[ui.name(k) for k in moved["market"]], orientation="h",
            marker_color=[fx.DIV_POS if v >= 0 else fx.DIV_NEG for v in moved["delta"]],
            text=[ui.signed(v) for v in moved["delta"]], textposition="outside", textfont={"color": fx.INK_2},
            hovertemplate="%{y}: %{x:+,.0f} guests<extra></extra>"))
        fx.layout(fig, "Change in hotel guests by nationality (plan year)", height=420, legend=False)
        lo_x, hi_x = min(moved["delta"].min(), 0.0), max(moved["delta"].max(), 0.0)
        span = max(hi_x - lo_x, 1.0)
        fig.update_xaxes(title="Hotel guests vs baseline", tickformat="+,.0f", zeroline=True, zerolinecolor=fx.AXIS,
                         showgrid=True, gridcolor=fx.GRID,
                         range=[lo_x - 0.22 * span * (lo_x < 0), hi_x + 0.22 * span * (hi_x > 0)])
        fig.update_layout(margin={"r": 40, "l": 40})
        ui.chart(fig, "sim_market")
        show = by.assign(name=[ui.name(k) for k in by["market"]]).sort_values("guests", ascending=False)
        st.dataframe(show[["name", "guests", "delta", "nights", "dn"]].rename(columns={
            "name": "Nationality", "guests": "Hotel guests", "delta": "Change in guests", "nights": "Guest nights",
            "dn": "Change in nights"}), hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%,.0f") for c in
                           ("Hotel guests", "Change in guests", "Guest nights", "Change in nights")})
    with tabs[2]:
        c1, c2 = st.columns(2)
        if c1.button("Save current as Scenario A", key="save_a"):
            st.session_state["scenario_a"] = sc.model_dump_json()
        if c2.button("Save current as Scenario B", key="save_b"):
            st.session_state["scenario_b"] = sc.model_dump_json()
        a, b = st.session_state.get("scenario_a"), st.session_state.get("scenario_b")
        if not (a and b):
            st.caption("Save two lever settings as A and B to compare them side by side.")
        else:
            ra, rb = run_cached(a), run_cached(b)
            ca, cb = st.columns(2)
            for col, r, lab in ((ca, ra, "A"), (cb, rb, "B")):
                col.markdown(f"**Scenario {lab}:** {describe(list(r.scenario.levers))}")
                tt = r.totals
                col.markdown(ui.kpi("Change in hotel guests", ui.signed(tt["delta_guests"]),
                                    f"80%: {ui.signed(tt['delta_guests_interval']['lo80'])} to "
                                    f"{ui.signed(tt['delta_guests_interval']['hi80'])}"), unsafe_allow_html=True)
                col.markdown(ui.kpi("Change in guest nights", ui.signed(tt["delta_nights"]), ""),
                             unsafe_allow_html=True)
            fig = go.Figure()
            for r, lab, c in ((ra, "A", fx.SERIES[0]), (rb, "B", fx.SERIES[1])):
                p_, lo, hi = monthly_delta(r, bl, "guests")
                fig.add_bar(x=r.monthly["month"], y=p_, name=f"Scenario {lab}", marker_color=c,
                            error_y={"type": "data", "symmetric": False, "array": hi - p_, "arrayminus": p_ - lo,
                                     "color": fx.INK_2, "thickness": 1})
            fig.update_layout(barmode="group")
            fx.layout(fig, "Scenario A vs B: change in hotel guests by month (80% range)", height=330)
            fig.update_yaxes(title="Hotel guests vs baseline", tickformat="+,.0f")
            fig.update_xaxes(dtick="M1", tickformat="%b")
            ui.chart(fig, "sim_ab")
    with tabs[3]:
        st.download_button("Download results (CSV)", res.table.to_csv(index=False).encode(), file_name=
                           f"{ui.product()}_scenario_{sc.key()}.csv", mime="text/csv", key="dl_csv")
        st.download_button("Download scenario (JSON)", sc.model_dump_json(indent=2).encode(),
                           file_name=f"{ui.product()}_scenario_{sc.key()}.json", mime="application/json",
                           key="dl_json")
        st.caption("The JSON file reproduces this scenario exactly (same levers, seed and number of draws).")


def render() -> None:
    e = ui.engine()
    init_state(e)
    ui.page_setup("Scenario simulator", f"Plan year built from the latest 12 months of schedule "
                  f"({e.months[0]:%b %Y} to {e.months[-1]:%b %Y}). Change a lever; guests and nights update "
                  "with 80% and 95% ranges.")
    left, right = st.columns([1, 2.35], gap="large")
    with left:
        lever_panel(e)
    with right:
        try:
            sc = Scenario(name="Scenario", levers=build_levers(e))
            res = run_cached(sc.model_dump_json())
        except (ScenarioError, ValueError) as err:
            st.error(f"This combination of levers is not possible: {err}")
            return
        results_panel(e, sc, res)
    ui.synthetic_note()


# expose presets for the video/deck so they match the app exactly
def preset_scenario(name: str, e) -> Scenario:
    p = preset_mod.presets(top_market=top_origin(e), new_route_origin="CA")
    return p[name]
