"""Derived airport-month table (load factor, transfer share) and the data-quality report."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from skystay import paths, settings
from skystay.io.adapters import AdapterLog

LF_CLIP = 1.05


def derive_airport_month(flights: pd.DataFrame, pax: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Load factor and transfer share at the finest grain both tables share (airport x month)."""
    seats = flights.groupby(["month", "origin_airport", "origin_country"], as_index=False)[["flights", "seats"]].sum()
    am = seats.merge(pax, on=["month", "origin_airport", "origin_country"], how="outer", indicator=True)
    no_pax = am[am["_merge"] == "left_only"]
    no_seats = am[am["_merge"] == "right_only"]
    am = am.drop(columns="_merge")
    am["lf_raw"] = am["pax_arriving"] / am["seats"].replace(0, np.nan)
    over = am["lf_raw"] > 1.0
    clipped = am["lf_raw"] > LF_CLIP
    am["lf_flag_over_1"] = over.fillna(False)
    am["pax_used"] = np.where(clipped.fillna(False), am["seats"] * LF_CLIP, am["pax_arriving"])
    am["load_factor"] = (am["pax_used"] / am["seats"].replace(0, np.nan)).clip(upper=LF_CLIP)
    am["transfer_share"] = am["pax_transfer"] / am["pax_arriving"].replace(0, np.nan)
    stats = {
        "airport_months": int(len(am)),
        "lf_above_1": int(over.sum()),
        "lf_clipped_at_1_05": int(clipped.sum()),
        "lf_above_1_rows": am.loc[over.fillna(False), ["month", "origin_airport", "lf_raw"]]
        .assign(month=lambda d: d["month"].dt.strftime("%Y-%m"), lf_raw=lambda d: d["lf_raw"].round(3))
        .to_dict("records"),
        "seats_without_pax": no_pax[["month", "origin_airport"]]
        .assign(month=lambda d: d["month"].dt.strftime("%Y-%m")).to_dict("records"),
        "pax_without_seats": no_seats[["month", "origin_airport"]]
        .assign(month=lambda d: d["month"].dt.strftime("%Y-%m")).to_dict("records"),
    }
    return am, stats


def _robust_outliers(df: pd.DataFrame, key: str, value: str, label: str, z: float = 4.0) -> list[dict]:
    out = []
    for k, g in df[df[value] > 0].groupby(key):
        x = np.log(g[value].astype(float))
        med = x.median()
        # floor the MAD (in log units) so small discrete schedule wobble is not flagged
        mad = max((x - med).abs().median() * 1.4826, 0.05)
        if len(g) < 12:
            continue
        rz = (x - med) / mad
        for idx in rz[rz.abs() > z].index:
            out.append({"series": label, "key": k, "month": g.loc[idx, "month"].strftime("%Y-%m"),
                        "value": float(g.loc[idx, value]), "robust_z": round(float(rz[idx]), 2)})
    return out


def _coverage(df: pd.DataFrame, col: str = "month") -> dict:
    m = df[col]
    return {"first": m.min().strftime("%Y-%m"), "last": m.max().strftime("%Y-%m"), "n_months": int(m.nunique())}


def _missing_months(df: pd.DataFrame, key: str, window: pd.DatetimeIndex) -> list[dict]:
    present = set(zip(df[key], df["month"], strict=True))
    out = []
    for k in sorted(df[key].unique()):
        miss = [m for m in window if (k, m) not in present]
        if miss:
            out.append({key: k, "missing": [m.strftime("%Y-%m") for m in miss]})
    return out


def report(tables: dict[str, pd.DataFrame], logs: list[AdapterLog], am_stats: dict) -> dict:
    fl, px, ho = tables["flights_month"], tables["pax_month"], tables["hotel_month"]
    months_f, months_p, months_h = set(fl["month"]), set(px["month"]), set(ho["month"])
    overlap = sorted(months_f & months_p & months_h)
    window = pd.date_range(overlap[0], overlap[-1], freq="MS") if overlap else pd.DatetimeIndex([])
    gaps_in_overlap = [m.strftime("%Y-%m") for m in window if m not in set(overlap)]

    # service starts / stops per origin country (legitimate gaps, listed for the team)
    svc = fl.groupby("origin_country")["month"].agg(["min", "max", "nunique"]).reset_index()
    svc_changes = [
        {"origin": r.origin_country, "first": r.min.strftime("%Y-%m"), "last": r.max.strftime("%Y-%m"), "months": int(r.nunique)}
        for r in svc.itertuples()
        if r.min > window.min() or r.max < window.max()
    ]
    am_seats = fl.groupby(["month", "origin_airport"], as_index=False)["seats"].sum()
    outliers = (
        _robust_outliers(ho, "nationality", "guest_arrivals", "hotel guests")
        + _robust_outliers(am_seats, "origin_airport", "seats", "airport seats")
        + _robust_outliers(px, "origin_airport", "pax_arriving", "airport pax")
    )
    n_overlap = len(window)
    min_bridge = settings.load_settings()["reconciliation"]["min_months_for_bridge"]
    rep = {
        "data_mode": settings.data_mode(),
        "adapters": [
            {"dataset": lg.dataset, "source": lg.source, "rows_in": lg.rows_in, "rows_out": lg.rows_out, "notes": lg.notes}
            for lg in logs
        ],
        "coverage": {
            "flights_month": _coverage(fl),
            "pax_month": _coverage(px),
            "hotel_month": _coverage(ho),
        },
        "overlap_window": {
            "first": window.min().strftime("%Y-%m") if n_overlap else None,
            "last": window.max().strftime("%Y-%m") if n_overlap else None,
            "n_months": n_overlap,
            "gaps": gaps_in_overlap,
        },
        "missing_hotel_months": _missing_months(ho, "nationality", window),
        "missing_pax_months": am_stats["seats_without_pax"],
        "pax_without_seats": am_stats["pax_without_seats"],
        "service_starts_and_stops": svc_changes,
        "load_factor": {
            "airport_months": am_stats["airport_months"],
            "above_1": am_stats["lf_above_1"],
            "clipped_at_1_05": am_stats["lf_clipped_at_1_05"],
            "rows_above_1": am_stats["lf_above_1_rows"],
        },
        "outliers_robust_z_gt_4": outliers,
        "counts": {
            "origin_countries": int(fl["origin_country"].nunique()),
            "origin_airports": int(fl["origin_airport"].nunique()),
            "nationalities": int(ho["nationality"].nunique()),
            "events": int(len(tables["events"])),
            "holiday_days": int(len(tables["holidays"])),
        },
        "reconciliation_feasibility": {
            "overlap_months": n_overlap,
            "L0_identity": True,
            "L1_regional": True,
            "L2_bridge": n_overlap >= min_bridge,
            "note": (
                f"{n_overlap} overlapping months >= {min_bridge}: L2 learned bridge is feasible"
                if n_overlap >= min_bridge
                else f"only {n_overlap} overlapping months (< {min_bridge}): stay at L1"
            ),
        },
    }
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS / "data_quality.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (paths.REPORTS / "data_quality.md").write_text(render_markdown(rep), encoding="utf-8")
    return rep


def render_markdown(rep: dict) -> str:
    syn = rep["data_mode"] == "synthetic"
    lines = ["# Data-quality report", ""]
    if syn:
        lines += ["> **SYNTHETIC DEMO DATA.** Generated stand-in with known ground truth; not real Abu Dhabi data.", ""]
    lines += ["## Ingestion", "", "| Dataset | Source | Rows in | Rows out | Notes |", "| :- | :- | -: | -: | :- |"]
    for a in rep["adapters"]:
        lines.append(f"| {a['dataset']} | {a['source']} | {a['rows_in']:,} | {a['rows_out']:,} | {'; '.join(a['notes']) or '-'} |")
    lines += ["", "## Coverage", "", "| Table | First | Last | Months |", "| :- | :- | :- | -: |"]
    for t, c in rep["coverage"].items():
        lines.append(f"| {t} | {c['first']} | {c['last']} | {c['n_months']} |")
    ow = rep["overlap_window"]
    lines += ["", f"**Overlap (modelling) window:** {ow['first']} to {ow['last']} ({ow['n_months']} months); gaps: {ow['gaps'] or 'none'}.", ""]
    lines += ["## Missing records", ""]
    for m in rep["missing_hotel_months"]:
        lines.append(f"- Hotel, nationality {m['nationality']}: missing {', '.join(m['missing'])}")
    for m in rep["missing_pax_months"]:
        lines.append(f"- Passenger traffic, airport {m['origin_airport']}: seats scheduled but no traffic row in {m['month']}")
    for m in rep["pax_without_seats"]:
        lines.append(f"- Passenger traffic, airport {m['origin_airport']}: traffic row but no scheduled seats in {m['month']}")
    if not (rep["missing_hotel_months"] or rep["missing_pax_months"] or rep["pax_without_seats"]):
        lines.append("- none")
    lines += ["", "## Service starts and stops (legitimate gaps, not defects)", ""]
    for s in rep["service_starts_and_stops"]:
        lines.append(f"- {s['origin']}: served {s['first']} to {s['last']} ({s['months']} months)")
    lf = rep["load_factor"]
    lines += ["", "## Load factor", "",
              f"{lf['airport_months']:,} airport-months; {lf['above_1']} with LF > 1.0 (flagged); {lf['clipped_at_1_05']} clipped at 1.05."]
    for r in lf["rows_above_1"]:
        lines.append(f"- {r['origin_airport']} {r['month']}: LF {r['lf_raw']}")
    lines += ["", "## Outliers (robust z > 4 on log values)", ""]
    if rep["outliers_robust_z_gt_4"]:
        lines += ["| Series | Key | Month | Value | Robust z |", "| :- | :- | :- | -: | -: |"]
        for o in rep["outliers_robust_z_gt_4"]:
            lines.append(f"| {o['series']} | {o['key']} | {o['month']} | {o['value']:,.0f} | {o['robust_z']} |")
    else:
        lines.append("none")
    rf = rep["reconciliation_feasibility"]
    lines += ["", "## Country reconciliation feasibility", "", f"- {rf['note']}", ""]
    return "\n".join(lines)
