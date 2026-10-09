"""Nationality x month feature frames shared by the econometric and ML layers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skystay.chain import parametric as par
from skystay.chain.panel import Panels


def frame(
    p: Panels,
    model: par.ParametricModel,
    months: pd.DatetimeIndex,
    use_actual_supply: bool,
    seats: pd.DataFrame | None = None,
    calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rows = nationality x month over `months`.

    Columns: guests/nights (actuals where known), the Layer-1 prediction, air
    visitors attributed to the nationality, calendar features, trend and the
    12-month lag of log guests (taken from the panel history, so it is always
    at or beyond the 12-month forecast horizon).
    `use_actual_supply`: observed terminating passengers (training rows) vs
    seats x LF x (1-TR) (forecast rows). `seats` / `calendar` override the
    panel's (used by the scenario engine).
    """
    seats = par.seats_frame(p, months) if seats is None else seats
    calendar = par.calendar_frame(p, months) if calendar is None else calendar
    term = par.terminating_frame(p, months) if use_actual_supply else None
    pred = par.predict(model, seats, term, calendar)
    df = pred.merge(calendar, on=["nationality", "month"], how="left")
    actual = p.nat_month[["nationality", "month", "guests", "nights"]]
    df = df.merge(actual, on=["nationality", "month"], how="left")
    hist = p.nat_month[["nationality", "month", "guests"]].copy()
    hist["month"] = hist["month"] + pd.DateOffset(months=12)
    hist = hist.rename(columns={"guests": "guests_lag12"})
    df = df.merge(hist, on=["nationality", "month"], how="left")
    df["moy"] = df["month"].dt.month
    df["hol_frac"] = df["holiday_days"] / df["dim"]
    df["evt_frac"] = df["event_days"] / df["dim"]
    df["trend"] = (df["month"].dt.year - p.months[0].year) * 12 + df["month"].dt.month - p.months[0].month
    df["log_air"] = np.log1p(df["air_visitors"].clip(lower=0))
    df["lag12"] = np.log1p(df["guests_lag12"])
    df["l1_guests"] = df["guests_pred"]
    df["l1_nights"] = df["nights_pred"]
    return df
