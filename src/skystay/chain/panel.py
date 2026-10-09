"""Modelling panels built from the canonical store.

- origin_month: supply side, one row per departure country x month
- nat_month:    demand side, one row per guest nationality x month
Both are full grids over the modelling window (missing values stay NaN; zero
seats means no service that month).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from skystay import settings
from skystay.io import calendars, store
from skystay.reconcile import regions


@dataclass
class Panels:
    months: pd.DatetimeIndex
    origins: list[str]
    nationalities: list[str]
    origin_month: pd.DataFrame
    nat_month: pd.DataFrame
    seasonality: np.ndarray  # (12,), mean 1
    routes: pd.DataFrame  # flights_month (route x month)
    events: pd.DataFrame
    holidays: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def origin_matrix(self, col: str, months: pd.DatetimeIndex | None = None) -> np.ndarray:
        """(O, T) matrix of an origin_month column over `months` (default: all)."""
        months = self.months if months is None else months
        p = self.origin_month.pivot(index="origin", columns="month", values=col)
        return p.reindex(index=self.origins, columns=months).to_numpy(dtype=float)

    def nat_matrix(self, col: str, months: pd.DatetimeIndex | None = None) -> np.ndarray:
        months = self.months if months is None else months
        p = self.nat_month.pivot(index="nationality", columns="month", values=col)
        return p.reindex(index=self.nationalities, columns=months).to_numpy(dtype=float)


def _order_nationalities(keys: list[str]) -> list[str]:
    special = [k for k in ("AE", "OTHER") if k in keys]
    return sorted(k for k in keys if k not in special) + special


def build(tables: dict[str, pd.DataFrame] | None = None) -> Panels:
    if tables is None:
        tables = {t: store.read(t) for t in ("flights_month", "airport_month", "hotel_month", "events",
                                              "holidays", "seasonality")}
    fl, am, ho = tables["flights_month"], tables["airport_month"], tables["hotel_month"]
    months = pd.date_range(max(fl["month"].min(), ho["month"].min()), min(fl["month"].max(), ho["month"].max()), freq="MS")
    reg = regions.region_of()

    # ---- supply: airport-month -> origin-month (pax NaN if any served airport lacks traffic)
    a = am[am["seats"].fillna(0) > 0].copy()
    a["pax_missing"] = a["pax_used"].isna()
    g = a.groupby(["origin_country", "month"]).agg(
        seats=("seats", "sum"), flights=("flights", "sum"), pax=("pax_used", "sum"),
        transfer=("pax_transfer", "sum"), pax_missing=("pax_missing", "any"),
    ).reset_index().rename(columns={"origin_country": "origin"})
    g.loc[g["pax_missing"], ["pax", "transfer"]] = np.nan
    origins = sorted(g["origin"].unique())
    grid = pd.MultiIndex.from_product([origins, months], names=["origin", "month"]).to_frame(index=False)
    om = grid.merge(g.drop(columns="pax_missing"), on=["origin", "month"], how="left")
    om[["seats", "flights"]] = om[["seats", "flights"]].fillna(0.0)
    served = om["seats"] > 0
    om.loc[~served, ["pax", "transfer"]] = 0.0
    om["lf"] = np.where(served, om["pax"] / om["seats"].where(served), np.nan)
    om["tr"] = np.where(served & (om["pax"] > 0), om["transfer"] / om["pax"].where(om["pax"] > 0), np.nan)
    om["terminating"] = om["pax"] - om["transfer"]
    om["region"] = om["origin"].map(reg)
    om["moy"] = om["month"].dt.month

    # ---- demand: nationality-month with calendar features
    nats = _order_nationalities(sorted(ho["nationality"].unique()))
    grid = pd.MultiIndex.from_product([nats, months], names=["nationality", "month"]).to_frame(index=False)
    nm = grid.merge(
        ho.rename(columns={"guest_arrivals": "guests", "guest_nights": "nights"})[
            ["nationality", "month", "guests", "nights", "alos"]],
        on=["nationality", "month"], how="left",
    )
    hd = calendars.holiday_days_by_month(tables["holidays"]).rename(columns={"country": "nationality"})
    ed = calendars.event_days_by_month(tables["events"])
    nm = nm.merge(hd, on=["nationality", "month"], how="left").merge(ed, on="month", how="left")
    nm[["holiday_days", "event_days"]] = nm[["holiday_days", "event_days"]].fillna(0.0)
    nm["dim"] = nm["month"].dt.days_in_month
    nm["moy"] = nm["month"].dt.month
    nm["season"] = nm["moy"].map(settings.season_of_month)
    nm["region"] = nm["nationality"].map(reg)

    s = tables["seasonality"].sort_values("month_of_year")["index"].to_numpy(dtype=float)
    s = s / s.mean()
    return Panels(
        months=months, origins=origins, nationalities=nats, origin_month=om, nat_month=nm, seasonality=s,
        routes=fl, events=tables["events"], holidays=tables["holidays"],
        meta={"first": months[0].strftime("%Y-%m"), "last": months[-1].strftime("%Y-%m")},
    )


def restrict(p: Panels, cutoff: pd.Timestamp) -> Panels:
    """Panels containing only months strictly before `cutoff` (training view; no leakage)."""
    months = p.months[p.months < cutoff]
    return Panels(
        months=months,
        origins=p.origins,
        nationalities=p.nationalities,
        origin_month=p.origin_month[p.origin_month["month"] < cutoff].copy(),
        nat_month=p.nat_month[p.nat_month["month"] < cutoff].copy(),
        seasonality=p.seasonality,
        routes=p.routes[p.routes["month"] < cutoff].copy(),
        events=p.events,
        holidays=p.holidays,
        meta={**p.meta, "cutoff": cutoff.strftime("%Y-%m")},
    )
