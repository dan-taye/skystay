"""Calendar helpers.

`synthetic_holidays` / `synthetic_events` build the stand-in calendars used in
synthetic mode. Moving-holiday anchor dates (Eid, Lunar New Year, Diwali) are
approximate public dates, used only to give the synthetic data realistic
movement between months. `holiday_days_by_month` / `event_days_by_month` turn
canonical calendar tables into model features in both modes.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

EID_FITR = {2018: "2018-06-15", 2019: "2019-06-04", 2020: "2020-05-24", 2021: "2021-05-13",
            2022: "2022-05-02", 2023: "2023-04-21", 2024: "2024-04-10", 2025: "2025-03-30",
            2026: "2026-03-20"}
EID_ADHA = {2018: "2018-08-21", 2019: "2019-08-11", 2020: "2020-07-31", 2021: "2021-07-20",
            2022: "2022-07-09", 2023: "2023-06-28", 2024: "2024-06-16", 2025: "2025-06-06",
            2026: "2026-05-27"}
LUNAR_NY = {2018: "2018-02-16", 2019: "2019-02-05", 2020: "2020-01-25", 2021: "2021-02-12",
            2022: "2022-02-01", 2023: "2023-01-22", 2024: "2024-02-10", 2025: "2025-01-29",
            2026: "2026-02-17"}
DIWALI = {2018: "2018-11-07", 2019: "2019-10-27", 2020: "2020-11-14", 2021: "2021-11-04",
          2022: "2022-10-24", 2023: "2023-11-12", 2024: "2024-11-01", 2025: "2025-10-20",
          2026: "2026-11-08"}

MUSLIM_MAJORITY = {"SA", "OM", "KW", "QA", "BH", "EG", "JO", "MA", "PK", "AE", "NG"}


def easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return date(year, month, day + 1)


def _span(start: date, days: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(days)]


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _country_holidays(iso2: str, year: int) -> list[tuple[date, str]]:
    out: list[tuple[date, str]] = []

    def add(name: str, start: date, days: int) -> None:
        out.extend((d, name) for d in _span(start, days))

    xmas = date(year, 12, 21)
    if iso2 in MUSLIM_MAJORITY:
        add("Eid al-Fitr break", _d(EID_FITR[year]) - timedelta(days=1), 4)
        add("Eid al-Adha break", _d(EID_ADHA[year]) - timedelta(days=1), 5)
    if iso2 in {"GB", "DE", "FR", "IT"}:
        add("Christmas school break", xmas, 11)
        add("New Year school break", date(year, 1, 1), 3)
        add("Easter school break", easter(year) - timedelta(days=7), 10)
        add("Autumn half-term", date(year, 10, 24), 9)
    if iso2 == "RU":
        add("New Year holidays", date(year, 1, 1), 8)
        add("May holidays", date(year, 5, 1), 10)
    if iso2 in {"CN", "SG"}:
        add("Lunar New Year", _d(LUNAR_NY[year]) - timedelta(days=1), 8 if iso2 == "CN" else 2)
    if iso2 == "CN":
        add("Golden Week", date(year, 10, 1), 7)
    if iso2 == "JP":
        add("New Year holidays", date(year, 1, 1), 3)
        add("Year-end holidays", date(year, 12, 29), 3)
        add("Golden Week", date(year, 4, 29), 7)
        add("Obon", date(year, 8, 13), 4)
    if iso2 == "IN":
        add("Diwali", _d(DIWALI[year]) - timedelta(days=2), 5)
        add("Christmas", date(year, 12, 25), 1)
    if iso2 == "LK":
        add("Sinhala and Tamil New Year", date(year, 4, 13), 2)
        add("Christmas", date(year, 12, 25), 1)
    if iso2 == "PH":
        add("Holy Week", easter(year) - timedelta(days=3), 4)
        add("Christmas season", date(year, 12, 24), 8)
    if iso2 == "TH":
        add("Songkran", date(year, 4, 13), 3)
    if iso2 in {"US", "CA"}:
        add("Spring break", date(year, 3, 10), 10)
        add("Winter holidays", xmas + timedelta(days=1), 11)
    if iso2 == "US":
        nov1 = date(year, 11, 1)
        thanksgiving = nov1 + timedelta(days=(3 - nov1.weekday()) % 7 + 21)
        add("Thanksgiving", thanksgiving, 4)
    if iso2 in {"AU", "ZA"}:
        add("Summer school holidays", date(year, 12, 20), 12)
        add("Summer school holidays", date(year, 1, 1), 28)
        add("Easter", easter(year) - timedelta(days=2), 4)
    if iso2 in {"KE", "NG"}:
        add("Christmas", date(year, 12, 24), 8)
        add("Easter", easter(year) - timedelta(days=2), 4)
    if iso2 == "BR":
        add("Carnival", easter(year) - timedelta(days=48), 5)
        add("Christmas and New Year", date(year, 12, 24), 8)
    if iso2 == "AE":
        add("National Day", date(year, 12, 2), 2)
    return out


def synthetic_holidays(countries: list[str], years: list[int]) -> pd.DataFrame:
    rows = []
    for iso2 in countries:
        for y in years:
            for d, name in _country_holidays(iso2, y):
                rows.append((d, iso2, name))
    df = pd.DataFrame(rows, columns=["date", "country", "name"]).drop_duplicates(["date", "country"])
    df["date"] = pd.to_datetime(df["date"])
    lo = pd.Timestamp(year=min(years), month=1, day=1)
    hi = pd.Timestamp(year=max(years), month=12, day=31)
    return df[(df["date"] >= lo) & (df["date"] <= hi)].sort_values(["country", "date"]).reset_index(drop=True)


def synthetic_events(years: list[int], rng: np.random.Generator) -> pd.DataFrame:
    """Generic Abu Dhabi event calendar (names are generic placeholders)."""
    templates = [
        ("Motorsport weekend", "sport", (11, 28), 4),
        ("Golf championship", "sport", (1, 18), 4),
        ("Winter music festival", "culture", (2, 8), 3),
        ("Major trade exhibition (spring)", "business", (3, 10), 4),
        ("Book fair", "culture", (4, 28), 7),
        ("Summer family festival", "family", (7, 10), 14),
        ("Major trade exhibition (autumn)", "business", (11, 3), 4),
        ("Heritage and food festival", "culture", (12, 5), 10),
    ]
    rows = []
    for y in years:
        for name, typ, (m, d), length in templates:
            shift = int(rng.integers(-9, 10))
            start = date(y, m, d) + timedelta(days=shift)
            rows.append((name, typ, start, start + timedelta(days=length - 1)))
    df = pd.DataFrame(rows, columns=["name", "type", "start", "end"])
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def holiday_days_by_month(holidays: pd.DataFrame) -> pd.DataFrame:
    """Count of holiday days per (country, month). Columns: country, month, holiday_days."""
    h = holidays.copy()
    h["month"] = h["date"].dt.to_period("M").dt.to_timestamp()
    out = h.drop_duplicates(["country", "date"]).groupby(["country", "month"]).size()
    return out.rename("holiday_days").reset_index()


def event_days_by_month(events: pd.DataFrame) -> pd.DataFrame:
    """Count of distinct event days per month in Abu Dhabi. Columns: month, event_days."""
    days: set[pd.Timestamp] = set()
    for s, e in zip(events["start"], events["end"], strict=True):
        for d in pd.date_range(s, e, freq="D"):
            days.add(d)
    if not days:
        return pd.DataFrame({"month": pd.Series(dtype="datetime64[ns]"), "event_days": pd.Series(dtype=int)})
    s = pd.Series(sorted(days))
    out = s.dt.to_period("M").dt.to_timestamp().value_counts().sort_index()
    return out.rename("event_days").rename_axis("month").reset_index()
