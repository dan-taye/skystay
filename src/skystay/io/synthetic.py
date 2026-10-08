"""Synthetic stand-in for real tourism data, with a known ground-truth chain.

Writes raw files (same layout the adapters expect, see config/schema_map.yaml)
plus ground_truth.json to data/synthetic/generated/. Deterministic (seed 42).
This is NOT evidence about Abu Dhabi: it exists to build and verify the
pipeline, and to test that the models recover known parameters.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from skystay import paths
from skystay.io import calendars
from skystay.reconcile import regions

DOW_ORDER = [0, 3, 5, 1, 4, 6, 2]  # which weekdays get the extra flights


def _ts(x) -> pd.Timestamp | None:
    if x is None:
        return None
    return pd.Timestamp(x if not isinstance(x, date) else x.isoformat())


def _norm_shape(shape: list[float]) -> np.ndarray:
    a = np.asarray(shape, dtype=float)
    return a / a.mean()


def load_config(path: Path | None = None) -> dict:
    path = path or (paths.SYNTH / "generator.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def generate(out_dir: Path | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    out_dir = out_dir or paths.SYNTH_GEN
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg["seed"]))

    months = pd.date_range(_ts(cfg["start"]), _ts(cfg["end"]), freq="MS")
    n_t = len(months)
    moy = months.month.to_numpy()
    years_since = (np.arange(n_t)) / 12.0
    dim = months.days_in_month.to_numpy()

    reg = regions.region_of()
    shapes = {k: _norm_shape(v) for k, v in cfg["shapes"].items()}
    profile = cfg["region_profile"]
    seas_index = _norm_shape(cfg["seasonality_index"])
    hubs = set(cfg["hubs"])

    origins = list(cfg["origins"].keys())
    nats = list(cfg["nationalities"].keys())
    o_idx = {o: i for i, o in enumerate(origins)}
    n_idx = {n: i for i, n in enumerate(nats)}

    def shape_of(iso2: str) -> np.ndarray:
        return shapes[profile[reg[iso2]]]

    # ---------------- schedule (route x month weekly frequency) ----------------
    seats_per = cfg["aircraft_seats"]
    shock = cfg["capacity_shock"]
    shock_mask = (months >= _ts(shock["start"])) & (months <= _ts(shock["end"]))
    up = cfg["upgauge"]
    g_lo, g_hi = cfg["uplifts"]["growth_per_year"]
    step_p = cfg["uplifts"]["freq_step_prob"]

    route_rows = []  # monthly route table for ground truth
    daily_frames = []
    for o in origins:
        prof = profile[reg[o]]
        for r in cfg["origins"][o]["routes"]:
            airport, carrier, aircraft, f0 = r[0], r[1], r[2], float(r[3])
            start = _ts(r[4]) if len(r) > 4 and r[4] else months[0]
            end = _ts(r[5]) if len(r) > 5 and r[5] else months[-1]
            growth = rng.uniform(g_lo, g_hi)
            steps = np.ones(n_t)
            level = 1.0
            for t in range(n_t):
                if moy[t] in (4, 11) and rng.random() < step_p:
                    level *= float(np.exp(rng.normal(0, 0.10)))
                steps[t] = level
            sched_seas = np.ones(n_t)
            if prof == "leisure":
                sched_seas[np.isin(moy, [11, 12, 1, 2, 3])] = 1.12
            else:
                sched_seas[np.isin(moy, [6, 7, 8])] = 1.10
            freq = f0 * (1 + growth) ** years_since * steps * sched_seas
            if o in shock["origins"]:
                freq = np.where(shock_mask, freq * shock["multiplier"], freq)
            active = (months >= start) & (months <= end)
            freq = np.where(active, np.maximum(1, np.round(freq)), 0).astype(int)
            ac = np.array([aircraft] * n_t, dtype=object)
            if (o, airport, carrier) == (up["origin"], up["airport"], up["carrier"]):
                ac[months >= _ts(up["start"])] = up["to"]
            for t in range(n_t):
                if freq[t] == 0:
                    continue
                days = pd.date_range(months[t], months[t] + pd.offsets.MonthEnd(0), freq="D")
                base, extra = divmod(int(freq[t]), 7)
                dow = days.dayofweek.to_numpy()
                fl = base + np.isin(dow, DOW_ORDER[:extra]).astype(int)
                keep = fl > 0
                spf = seats_per[ac[t]]
                daily_frames.append(
                    pd.DataFrame(
                        {
                            "FLT_DATE": days[keep].strftime("%Y-%m-%d"),
                            "ORIGIN_IATA": airport,
                            "ORIGIN_COUNTRY": regions.label(o),
                            "CARRIER": f"{carrier}" if carrier != "FC" else f"FC-{o}",
                            "EQUIPMENT": ac[t],
                            "DEPARTURES": fl[keep],
                            "SEATS": fl[keep] * spf,
                        }
                    )
                )
                route_rows.append(
                    (months[t], o, airport, carrier, ac[t], int(freq[t]), int(fl.sum()), int(fl.sum() * spf))
                )
    schedule = pd.concat(daily_frames, ignore_index=True)
    routes_m = pd.DataFrame(
        route_rows, columns=["month", "origin", "airport", "carrier", "aircraft", "weekly_freq", "flights", "seats"]
    )

    # ---------------- load factor, transfers, passengers ----------------
    nz = cfg["noise"]
    lf_true = np.zeros((len(origins), n_t))
    tr_true = np.zeros((len(origins), n_t))
    for o in origins:
        i = o_idx[o]
        p = cfg["origins"][o]
        lf_true[i] = np.minimum(0.98, p["lf"] * shape_of(o)[moy - 1] * np.exp(rng.normal(0, nz["lf_sd"], n_t)))
        tr_true[i] = np.clip(
            p["tr"] * (1 + 0.08 * np.sin(2 * np.pi * moy / 12)) * np.exp(rng.normal(0, nz["tr_sd"], n_t)), 0, 0.9
        )
    airport_m = routes_m.groupby(["month", "origin", "airport"], as_index=False)["seats"].sum()
    t_index = {m: t for t, m in enumerate(months)}
    ti = airport_m["month"].map(t_index).to_numpy()
    oi = airport_m["origin"].map(o_idx).to_numpy()
    pax = airport_m["seats"].to_numpy() * lf_true[oi, ti] * np.exp(rng.normal(0, nz["pax_route_sd"], len(airport_m)))
    airport_m["pax"] = np.round(pax).astype(int)
    airport_m["transfer"] = np.round(airport_m["pax"] * tr_true[oi, ti]).astype(int)
    term = np.zeros((len(origins), n_t))
    np.add.at(term, (oi, ti), (airport_m["pax"] - airport_m["transfer"]).to_numpy())
    seats_o = np.zeros((len(origins), n_t))
    np.add.at(seats_o, (oi, ti), airport_m["seats"].to_numpy())
    res = np.array([cfg["origins"][o]["res"] for o in origins])
    visitors = term * (1 - res)[:, None]

    # ---------------- bridge B[o, n] ----------------
    bc = cfg["bridge"]
    B = np.zeros((len(origins), len(nats)))
    for o in origins:
        i = o_idx[o]
        lo, hi = bc["diag_hub"] if o in hubs else bc["diag_nonhub"]
        d = rng.uniform(lo, hi)
        B[i, n_idx[o]] = d
        B[i, n_idx["OTHER"]] = bc["other_share"]
        rest = 1 - d - bc["other_share"]
        if o in hubs:
            targets = [n for n in bc["hub_targets"] if n != o]
        else:
            targets = [n for n in nats if n not in ("OTHER", "AE", o) and reg.get(n) == reg[o]]
        if not targets:
            B[i, n_idx["OTHER"]] += rest
            continue
        w = rng.dirichlet(np.ones(len(targets)) * 1.5)
        for n, wn in zip(targets, w, strict=True):
            B[i, n_idx[n]] += rest * wn

    # ---------------- calendars ----------------
    years = sorted(set(months.year))
    holidays = calendars.synthetic_holidays(nats, years)
    events = calendars.synthetic_events(years, rng)
    hd = calendars.holiday_days_by_month(holidays)
    hol_days = np.zeros((len(nats), n_t))
    for row in hd.itertuples():
        if row.country in n_idx and row.month in t_index:
            hol_days[n_idx[row.country], t_index[row.month]] = row.holiday_days
    ed = calendars.event_days_by_month(events)
    ev_days = np.zeros(n_t)
    for row in ed.itertuples():
        if row.month in t_index:
            ev_days[t_index[row.month]] = row.event_days

    # ---------------- hotel side ----------------
    upl = cfg["uplifts"]
    hol_up = 1 + upl["holiday_capture"] * hol_days / dim
    ev_up = 1 + upl["event_capture"] * ev_days / dim
    cap = np.array([cfg["nationalities"][n]["cap"] for n in nats])
    amp = np.array([cfg["nationalities"][n]["cap_amp"] for n in nats])
    base = np.array([cfg["nationalities"][n]["base"] for n in nats], dtype=float)
    alos = np.array([cfg["nationalities"][n]["alos"] for n in nats])
    cap_seas = np.zeros((len(nats), n_t))
    alos_seas = np.zeros((len(nats), n_t))
    for n in nats:
        s = shape_of(n)
        z = (s - 1) / np.max(np.abs(s - 1))
        cap_seas[n_idx[n]] = 1 + amp[n_idx[n]] * z[moy - 1]
        alos_seas[n_idx[n]] = 1 + 0.06 * z[moy - 1]
    flow = B[:, :, None] * visitors[:, None, :]  # o x n x t
    capture_t = cap[:, None] * cap_seas * hol_up * ev_up[None, :]  # n x t
    air_flow = flow * capture_t[None, :, :]  # o x n x t
    air = air_flow.sum(axis=0)
    base_t = base[:, None] * seas_index[moy - 1][None, :] * (1 + upl["base_trend_per_year"]) ** years_since * ev_up
    guests = (air + base_t) * np.exp(rng.normal(0, nz["guests_sd"], air.shape))
    guests = np.round(guests).astype(int)
    alos_t = alos[:, None] * alos_seas
    nights = np.round(guests * alos_t * np.exp(rng.normal(0, nz["nights_sd"], air.shape))).astype(int)

    rm = cfg["rooms"]
    rooms = rm["start"] * (1 + rm["growth_per_year"]) ** years_since
    occ = nights.sum(axis=0) / rm["guests_per_room"] / (rooms * dim) * 100

    # ---------------- raw files ----------------
    hotel_label = {n: regions.label(n) for n in nats}
    hotel_label.update({"GB": "UK", "US": "USA", "AE": "UAE", "OTHER": "Others", "KR": "Korea, Republic of"})
    hotel_rows = []
    for n in nats:
        for t in range(n_t):
            g = int(guests[n_idx[n], t])
            ni = int(nights[n_idx[n], t])
            hotel_rows.append(
                (months[t].strftime("%Y-%m"), hotel_label[n], g, ni, round(ni / g, 2) if g else None, round(occ[t], 1))
            )
    hotel = pd.DataFrame(
        hotel_rows, columns=["Month", "Nationality", "Guest Arrivals", "Guest Nights", "ALOS", "Occupancy %"]
    )
    pax_df = pd.DataFrame(
        {
            "PERIOD": airport_m["month"].dt.strftime("%Y-%m"),
            "ORIGIN_IATA": airport_m["airport"],
            "ORIGIN_COUNTRY": airport_m["origin"].map(regions.label),
            "ARR_PAX": airport_m["pax"],
            "TRANSFER_PAX": airport_m["transfer"],
        }
    )

    # defects (documented in generator.yaml)
    df_cfg = cfg["defects"]
    for m, n in df_cfg["drop_hotel_rows"]:
        key = (_ts(m).strftime("%Y-%m"), hotel_label[n])
        hotel = hotel[~((hotel["Month"] == key[0]) & (hotel["Nationality"] == key[1]))]
    for m, n in df_cfg["duplicate_hotel_rows"]:
        key = (_ts(m).strftime("%Y-%m"), hotel_label[n])
        dup = hotel[(hotel["Month"] == key[0]) & (hotel["Nationality"] == key[1])]
        hotel = pd.concat([hotel, dup], ignore_index=True)
    for m, ap in df_cfg["drop_pax_rows"]:
        pax_df = pax_df[~((pax_df["PERIOD"] == _ts(m).strftime("%Y-%m")) & (pax_df["ORIGIN_IATA"] == ap))]
    for m, ap, ratio in df_cfg["overfull_pax_rows"]:
        mask = (pax_df["PERIOD"] == _ts(m).strftime("%Y-%m")) & (pax_df["ORIGIN_IATA"] == ap)
        seats_ap = airport_m.loc[
            (airport_m["month"] == _ts(m)) & (airport_m["airport"] == ap), "seats"
        ].sum()
        pax_df.loc[mask, "ARR_PAX"] = int(seats_ap * ratio)

    seasonality = pd.DataFrame({"Month": range(1, 13), "Index": np.round(seas_index, 3)})
    hol_out = holidays.assign(country=holidays["country"].map(regions.label))
    hol_out = hol_out.rename(columns={"date": "Date", "country": "Country", "name": "Holiday"})
    hol_out["Date"] = hol_out["Date"].dt.strftime("%Y-%m-%d")
    ev_out = events.rename(columns={"name": "Event", "type": "Type", "start": "Start Date", "end": "End Date"})
    ev_out["Start Date"] = ev_out["Start Date"].dt.strftime("%Y-%m-%d")
    ev_out["End Date"] = ev_out["End Date"].dt.strftime("%Y-%m-%d")

    schedule.to_csv(out_dir / "flight_schedule.csv", index=False)
    pax_df.to_csv(out_dir / "pax_traffic.csv", index=False)
    hotel.to_csv(out_dir / "hotel_performance.csv", index=False)
    ev_out.to_csv(out_dir / "events.csv", index=False)
    hol_out.to_csv(out_dir / "holidays.csv", index=False)
    seasonality.to_csv(out_dir / "seasonality.csv", index=False)
    _write_dictionary(out_dir)

    # ---------------- ground truth ----------------
    k_eff = air_flow.sum(axis=2) / np.maximum(term.sum(axis=1), 1)[:, None]  # guests per terminating pax
    mix = k_eff / np.maximum(k_eff.sum(axis=1, keepdims=True), 1e-12)
    air_tot = air.sum(axis=1)
    base_tot = base_t.sum(axis=1)
    truth = {
        "note": "Synthetic ground truth. Not real data.",
        "seed": int(cfg["seed"]),
        "months": [months[0].strftime("%Y-%m"), months[-1].strftime("%Y-%m")],
        "origins": origins,
        "nationalities": nats,
        "load_factor": {o: cfg["origins"][o]["lf"] for o in origins},
        "load_factor_realised": {
            o: float((lf_true[o_idx[o]] * seats_o[o_idx[o]]).sum() / max(seats_o[o_idx[o]].sum(), 1)) for o in origins
        },
        "transfer_share": {o: cfg["origins"][o]["tr"] for o in origins},
        "returning_resident_share": {o: cfg["origins"][o]["res"] for o in origins},
        "capture_rate": {n: cfg["nationalities"][n]["cap"] for n in nats},
        "alos": {n: cfg["nationalities"][n]["alos"] for n in nats},
        "non_air_base_per_month": {n: cfg["nationalities"][n]["base"] for n in nats},
        "non_air_share": {n: float(base_tot[n_idx[n]] / (air_tot[n_idx[n]] + base_tot[n_idx[n]])) for n in nats},
        "bridge_B": {o: {n: round(float(B[o_idx[o], n_idx[n]]), 6) for n in nats if B[o_idx[o], n_idx[n]] > 0} for o in origins},
        "effective_K": {o: {n: float(k_eff[o_idx[o], n_idx[n]]) for n in nats if k_eff[o_idx[o], n_idx[n]] > 0} for o in origins},
        "mix_diagonal": {o: float(mix[o_idx[o], n_idx[o]]) for o in origins},
        "effective_capture_diagonal": {o: float(k_eff[o_idx[o], n_idx[o]]) for o in origins},
        "uplifts": cfg["uplifts"],
        "route_events": {
            "launches": {"BR": "2021-07", "KE": "2022-10", "JP": "2023-04"},
            "cancellations": {"MA": "2021-12 (last month)", "NG": "2023-06 (last month)"},
            "capacity_shock": {"origins": shock["origins"], "months": "2021-05..2021-07", "multiplier": shock["multiplier"]},
        },
        "defects": {k: [[str(x) for x in row] for row in v] for k, v in cfg["defects"].items()},
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(
        f"synthetic: {len(schedule):,} schedule rows, {len(pax_df):,} pax rows, {len(hotel):,} hotel rows, "
        f"{len(ev_out)} events, {len(hol_out):,} holiday-days -> {out_dir}"
    )
    return truth


def _write_dictionary(out_dir: Path) -> None:
    rows = [
        ("flight_schedule.csv", "FLT_DATE", "date", "Flight date (local), one row per origin airport x carrier x aircraft x day"),
        ("flight_schedule.csv", "ORIGIN_IATA", "text", "Departure airport IATA code"),
        ("flight_schedule.csv", "ORIGIN_COUNTRY", "text", "Departure country (NOT passenger nationality)"),
        ("flight_schedule.csv", "CARRIER", "text", "Carrier code (synthetic: HC home network, HL home low-cost, FC-xx foreign)"),
        ("flight_schedule.csv", "EQUIPMENT", "text", "Aircraft type"),
        ("flight_schedule.csv", "DEPARTURES", "int", "Scheduled arriving flights to Abu Dhabi that day"),
        ("flight_schedule.csv", "SEATS", "int", "Scheduled seats on those flights"),
        ("pax_traffic.csv", "PERIOD", "YYYY-MM", "Month"),
        ("pax_traffic.csv", "ORIGIN_IATA", "text", "Departure airport IATA code"),
        ("pax_traffic.csv", "ORIGIN_COUNTRY", "text", "Departure country"),
        ("pax_traffic.csv", "ARR_PAX", "int", "Passengers arriving at Abu Dhabi from that airport"),
        ("pax_traffic.csv", "TRANSFER_PAX", "int", "Of which transferring/transiting onward (connection indicator)"),
        ("hotel_performance.csv", "Month", "YYYY-MM", "Month"),
        ("hotel_performance.csv", "Nationality", "text", "Guest nationality (NOT departure country); includes UAE and Others"),
        ("hotel_performance.csv", "Guest Arrivals", "int", "Hotel guest arrivals (check-ins)"),
        ("hotel_performance.csv", "Guest Nights", "int", "Hotel guest nights"),
        ("hotel_performance.csv", "ALOS", "float", "Average length of stay = nights / arrivals"),
        ("hotel_performance.csv", "Occupancy %", "float", "All-hotel room occupancy for the month (same value on every row)"),
        ("events.csv", "Event/Type/Start Date/End Date", "text/date", "Abu Dhabi event calendar (generic names)"),
        ("holidays.csv", "Date/Country/Holiday", "date/text", "Source-market holiday days (approximate dates)"),
        ("seasonality.csv", "Month/Index", "int/float", "Abu Dhabi seasonality profile, mean 1"),
    ]
    pd.DataFrame(rows, columns=["file", "column", "type", "description"]).to_csv(
        out_dir / "data_dictionary.csv", index=False
    )
