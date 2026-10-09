"""Generated assumption ledger: config/assumptions.yaml and docs/ASSUMPTIONS.md.

Every assumption carries value, unit, 80% interval, n_obs, method, source and
an adjustable flag. Intervals are the uncertainty of the ESTIMATE (empirical-
Bayes posterior or year-block bootstrap), not month-to-month variation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import yaml

from skystay import paths, settings
from skystay.chain.shrink import Z80
from skystay.reconcile import regions

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def _moy_entry(value: np.ndarray, se: np.ndarray, n_obs: int, unit: str, method: str, source: str,
               lo_clip=0.0, hi_clip=None) -> dict:
    lo = np.clip(value - Z80 * se, lo_clip, hi_clip)
    hi = np.clip(value + Z80 * se, lo_clip, hi_clip)
    return {
        "value": _r(value.mean()),
        "unit": unit,
        "interval_80": [_r(lo.mean()), _r(hi.mean())],
        "n_obs": int(n_obs),
        "method": method,
        "source": source,
        "adjustable": True,
        "by_month": {MONTHS[k]: {"value": _r(value[k]), "interval_80": [_r(lo[k]), _r(hi[k])]} for k in range(12)},
    }


def build(b, summary: dict) -> dict:
    m, p = b.chain, b.panels
    syn = settings.is_synthetic()
    src_data = "synthetic stand-in data (data/synthetic)" if syn else "real data (data/raw)"
    om, nm = p.origin_month, p.nat_month
    out: dict = {
        "meta": {
            "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
            "generated_by": "skystay train (src/skystay/chain/assumptions.py) - do not edit by hand",
            "data_mode": settings.data_mode(),
            "fitted_on": b.meta["fitted_on"],
            "reconciliation_level": b.level,
            "model": b.model_name,
            "interval_meaning": "80% interval of the estimate (EB posterior or year-block bootstrap), "
                                "not month-to-month variation",
        },
        "load_factor": {},
        "transfer_share": {},
        "length_of_stay": {},
        "capture_rate": {},
        "non_air_baseline": {},
    }
    for i, o in enumerate(m.origins):
        n_obs = int((om[(om["origin"] == o)]["seats"] > 0).sum())
        out["load_factor"][o] = _moy_entry(
            m.lf[i], m.lf_se[i], n_obs, "passengers / seats", "empirical-Bayes month-of-year means shrunk to the "
            "market's annual mean (thin markets shrunk to region)", src_data, 0.0, 1.05)
        out["transfer_share"][o] = _moy_entry(
            m.tr[i], m.tr_se[i], n_obs, "transfer passengers / arriving passengers",
            "EB month-of-year means of the traffic data's transfer indicator", src_data, 0.0, 1.0)
    se_rel = summary.get("capture_se_rel", {})
    for j, n in enumerate(m.nationalities):
        n_obs = int(nm[(nm["nationality"] == n)]["guests"].notna().sum())
        out["length_of_stay"][n] = _moy_entry(m.alos[j], m.alos_se[j], n_obs, "nights per guest",
                                              "EB month-of-year means of nights / arrivals", src_data, 0.0)
        cap = summary["capture_per_visitor"].get(n)
        if cap is not None and np.isfinite(cap):
            sr = se_rel.get(n, 0.0)
            out["capture_rate"][n] = {
                "value": _r(cap), "unit": "air-attributable hotel guests per terminating passenger (returning "
                "residents folded in)", "interval_80": [_r(cap * max(1 - Z80 * sr, 0)), _r(cap * (1 + Z80 * sr))],
                "n_obs": n_obs, "method": f"{b.level} conversion fitted by bounded least squares; interval from a "
                "year-block bootstrap of the capture level", "source": src_data, "adjustable": True,
            }
        base_month = float(m.base[j].mean())
        share = summary["non_air_share"].get(n)
        bse = 0.0
        if m.unit_base_se_rel is not None:
            units = m.units
            u = n if n in units else regions.region_of()[n]
            if u in units:
                bse = float(m.unit_base_se_rel[units.index(u)])
        out["non_air_baseline"][n] = {
            "value": _r(base_month, 1), "unit": "hotel guests per month not attributable to flights into Abu Dhabi "
            "(domestic, land, other UAE gateways), level at the end of the data",
            "interval_80": [_r(base_month * max(1 - Z80 * bse, 0), 1), _r(base_month * (1 + Z80 * bse), 1)],
            "non_air_share_last12": _r(share), "n_obs": n_obs,
            "method": "intercept of the conversion regression, shaped by the DCT seasonality profile, with a "
                      "ridge-penalised trend; year-block bootstrap interval",
            "source": src_data, "adjustable": False,
            "note": "Scenario levers move only the air-attributable part.",
        }
    cd = m.calendar_detail.get("unit_estimates") or []
    arr = np.array(cd) if cd else np.zeros((0, 3))

    def pct(k):
        return [_r(np.percentile(arr[:, k], 10)), _r(np.percentile(arr[:, k], 90))] if len(arr) else [None, None]

    out["calendar"] = {
        "holiday_uplift": {"value": _r(m.hol_uplift), "unit": "relative capture uplift per (holiday days / days in "
                           "month) in the source market", "interval_80": pct(0), "n_obs": len(arr),
                           "method": "guest-weighted mean of regional estimates (bounded >= 0)", "source": src_data,
                           "adjustable": False},
        "event_uplift_air": {"value": _r(m.event_uplift_air), "unit": "relative uplift of air guests per (event days "
                             "/ days in month)", "interval_80": pct(1), "n_obs": len(arr),
                             "method": "as above", "source": src_data, "adjustable": True},
        "event_uplift_non_air": {"value": _r(m.event_uplift_base), "unit": "relative uplift of non-air guests per "
                                 "(event days / days in month)", "interval_80": pct(2), "n_obs": len(arr),
                                 "method": "as above", "source": src_data, "adjustable": True},
    }
    e = b.econ["elasticity"]
    out["structural"] = {
        "returning_residents": {"value": None, "unit": "share of terminating passengers",
                                "method": "not observable in aggregated traffic; folded into the capture rate",
                                "source": "modelling decision", "adjustable": False, "interval_80": [None, None],
                                "n_obs": 0},
        "reconciliation_level": {"value": b.level, "unit": "-", "method": "chosen by rolling-origin back-test "
                                 "(reports/metrics.json)", "source": "back-test", "adjustable": False,
                                 "interval_80": [None, None], "n_obs": 0},
        "bridge_ridge_lambda": {"value": summary["bridge"].get("lambda"), "unit": "-", "method": "chosen on the last "
                                "12 training months", "source": "back-test", "adjustable": False,
                                "interval_80": [None, None], "n_obs": 0},
        "cold_start_rule": {"value": "region-average LF, transfer share and conversion; nationality = origin "
                            "country (else Others); uncertainty doubled", "unit": "-", "method": "rule",
                            "source": "modelling decision", "adjustable": False, "interval_80": [None, None],
                            "n_obs": 0},
        "elasticity_check": {"value": _r(e["coef"]), "unit": "% change in guests per % change in air visitors",
                             "interval_80": None, "ci95": [_r(e["ci95"][0]), _r(e["ci95"][1])],
                             "n_obs": b.econ["n_obs"], "method": "econometric layer, OLS with HC3 errors",
                             "source": src_data, "adjustable": False},
    }
    return out


def write(b, summary: dict) -> dict:
    data = build(b, summary)
    header = ("# GENERATED by `skystay train` - do not edit by hand.\n"
              "# Every assumption: value, unit, 80% interval of the estimate, n_obs, method, source, adjustable.\n")
    if settings.is_synthetic():
        header += "# SYNTHETIC DEMO DATA: values describe the synthetic stand-in, not Abu Dhabi.\n"
    paths.ASSUMPTIONS_FILE.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True,
                                                              width=120), encoding="utf-8")
    (paths.DOCS / "ASSUMPTIONS.md").write_text(markdown(data), encoding="utf-8")
    return data


def markdown(d: dict) -> str:
    syn = d["meta"]["data_mode"] == "synthetic"
    L = ["# Assumption ledger", "",
         "_Generated from `config/assumptions.yaml` by `skystay train`. Every value is estimated from the data "
         "unless marked as a rule or decision. Intervals are 80% intervals of the estimate._", ""]
    if syn:
        L += ["> **SYNTHETIC DEMO DATA.** These values describe the synthetic stand-in, not Abu Dhabi.", ""]
    meta = d["meta"]
    L += [f"Fitted on {meta['fitted_on']}; reconciliation level **{meta['reconciliation_level']}**; model "
          f"**{meta['model']}**.", ""]

    def table(title, fam, keyname):
        rows = [f"## {title}", "", f"Unit: {next(iter(fam.values()))['unit']}. Method: "
                f"{next(iter(fam.values()))['method']}.", "",
                f"| {keyname} | Value | 80% interval | n_obs | Adjustable |", "| :- | -: | :- | -: | :- |"]
        for k, v in fam.items():
            iv = v["interval_80"]
            rows.append(f"| {regions.label(k)} ({k}) | {v['value']} | {iv[0]} to {iv[1]} | {v['n_obs']} | "
                        f"{'yes' if v['adjustable'] else 'no'} |")
        return rows + [""]

    L += table("Load factor", d["load_factor"], "Origin")
    L += table("Transfer / transit share", d["transfer_share"], "Origin")
    L += table("Capture rate", d["capture_rate"], "Nationality")
    L += table("Average length of stay", d["length_of_stay"], "Nationality")
    L += ["## Non-air baseline", "", "Hotel guests not attributable to flights into Abu Dhabi (UAE residents, land "
          "arrivals, other UAE airports). Estimated from data; scenario levers do not move it.", "",
          "| Nationality | Guests / month | 80% interval | Non-air share (last 12 m) |", "| :- | -: | :- | -: |"]
    for k, v in d["non_air_baseline"].items():
        iv = v["interval_80"]
        sh = v["non_air_share_last12"]
        L.append(f"| {regions.label(k)} ({k}) | {v['value']:,.0f} | {iv[0]:,.0f} to {iv[1]:,.0f} | "
                 f"{'' if sh is None else f'{sh:.0%}'} |")
    L += ["", "## Calendar effects", "", "| Assumption | Value | 80% range across regions | Unit |",
          "| :- | -: | :- | :- |"]
    for k, v in d["calendar"].items():
        L.append(f"| {k} | {v['value']} | {v['interval_80'][0]} to {v['interval_80'][1]} | {v['unit']} |")
    L += ["", "## Structural assumptions and rules", "", "| Assumption | Value | Method / source |", "| :- | :- | :- |"]
    for k, v in d["structural"].items():
        val = v["value"]
        if k == "elasticity_check":
            val = f"{v['value']} (95% CI {v['ci95'][0]} to {v['ci95'][1]})"
        L.append(f"| {k} | {val} | {v['method']} ({v['source']}) |")
    return "\n".join(L) + "\n"
