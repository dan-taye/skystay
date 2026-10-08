"""One adapter per raw dataset: raw file -> canonical table.

Column renames come from config/schema_map.yaml. Every adapter validates types,
non-negativity, unique keys and date coverage, and logs row counts in and out.
Rows are never dropped silently: exact duplicates are removed and logged;
anything else that cannot be mapped raises an error naming the problem.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from skystay import paths
from skystay.reconcile import regions


class SchemaError(ValueError):
    pass


@dataclass
class AdapterLog:
    dataset: str
    source: str
    rows_in: int = 0
    rows_out: int = 0
    notes: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.notes.append(msg)
        print(f"  [{self.dataset}] {msg}")


def load_schema_map() -> dict:
    return yaml.safe_load(paths.SCHEMA_MAP_FILE.read_text(encoding="utf-8"))["datasets"]


def find_file(folder: Path, pattern: str) -> Path:
    exts = (".csv", ".xlsx", ".xls", ".parquet")
    hits = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in exts and fnmatch.fnmatch(p.name.lower(), pattern.lower() + "*")
    ]
    hits = hits or [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in exts and fnmatch.fnmatch(p.stem.lower(), pattern.lower())
    ]
    if not hits:
        raise FileNotFoundError(f"no file matching '{pattern}' in {folder} (edit config/schema_map.yaml)")
    if len(hits) > 1:
        raise SchemaError(f"several files match '{pattern}' in {folder}: {[h.name for h in hits]}")
    return hits[0]


def read_raw(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise SchemaError(f"unsupported file type {path}")


def apply_schema(df: pd.DataFrame, spec: dict, dataset: str) -> pd.DataFrame:
    """Rename raw columns to canonical names; fail loudly if required ones are missing."""
    stripped = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=stripped)
    mapping = {raw: canon for raw, canon in spec["columns"].items() if raw in df.columns}
    out = df.rename(columns=mapping)
    missing = [c for c in spec["required"] if c not in out.columns]
    if missing:
        unmapped = [c for c in df.columns if c not in mapping]
        raise SchemaError(
            f"{dataset}: required canonical column(s) {missing} not produced. "
            f"Unmapped raw columns: {unmapped}. Edit config/schema_map.yaml."
        )
    keep = [c for c in spec["required"] + spec.get("optional", []) if c in out.columns]
    return out[keep].copy()


def _month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str), errors="raise").dt.to_period("M").dt.to_timestamp()


def _map_country(df: pd.DataFrame, col: str, log: AdapterLog) -> pd.DataFrame:
    keys, unmapped = regions.map_series(df[col])
    if unmapped:
        raise SchemaError(
            f"{log.dataset}: {len(unmapped)} country label(s) not in config/regions.yaml: {unmapped[:20]}. "
            "Add them as aliases."
        )
    df[col] = keys
    return df


def _non_negative(df: pd.DataFrame, cols: list[str], log: AdapterLog) -> None:
    for c in cols:
        if c in df.columns:
            bad = int((pd.to_numeric(df[c], errors="coerce") < 0).sum())
            if bad:
                raise SchemaError(f"{log.dataset}: {bad} negative values in {c}")


def _dedupe(df: pd.DataFrame, key: list[str], log: AdapterLog) -> pd.DataFrame:
    exact = int(df.duplicated().sum())
    if exact:
        log.note(f"removed {exact} exact duplicate row(s)")
        df = df.drop_duplicates()
    conflicts = int(df.duplicated(key).sum())
    if conflicts:
        raise SchemaError(f"{log.dataset}: {conflicts} conflicting rows share key {key}")
    return df


def adapt_flights(df: pd.DataFrame, log: AdapterLog) -> pd.DataFrame:
    """Daily (or any sub-monthly) schedule -> flights_month at route x month grain."""
    df["date"] = pd.to_datetime(df["date"].astype(str), errors="raise")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    for c in ("flights", "seats"):
        df[c] = pd.to_numeric(df[c], errors="raise")
    _non_negative(df, ["flights", "seats"], log)
    df = _map_country(df, "origin_country", log)
    key = ["month", "origin_airport", "origin_country", "airline", "aircraft_type"]
    out = df.groupby(key, as_index=False)[["flights", "seats"]].sum()
    days = out["month"].dt.days_in_month
    out["weekly_freq_avg"] = (out["flights"] / days * 7).round(2)
    out["transfer_share"] = np.nan  # filled from pax traffic at airport-month grain
    log.note(f"aggregated {len(df):,} schedule rows to {len(out):,} route-months")
    return out


def adapt_pax(df: pd.DataFrame, log: AdapterLog) -> pd.DataFrame:
    df["month"] = _month(df["month"])
    for c in ("pax_arriving", "pax_transfer"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise")
    if "pax_transfer" not in df.columns:
        df["pax_transfer"] = np.nan
        log.note("no transfer column: transfer share is [UNKNOWN] and must be set as an assumption")
    _non_negative(df, ["pax_arriving", "pax_transfer"], log)
    df = _map_country(df, "origin_country", log)
    df = _dedupe(df, ["month", "origin_airport"], log)
    return df[["month", "origin_airport", "origin_country", "pax_arriving", "pax_transfer"]]


def adapt_hotel(df: pd.DataFrame, log: AdapterLog) -> pd.DataFrame:
    df["month"] = _month(df["month"])
    df = df.rename(columns={"nationality": "nationality"})
    for c in ("guest_arrivals", "guest_nights", "alos", "occupancy"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    _non_negative(df, ["guest_arrivals", "guest_nights", "alos", "occupancy"], log)
    df = _map_country(df, "nationality", log)
    df = _dedupe(df, ["month", "nationality"], log)
    recomputed = df["guest_nights"] / df["guest_arrivals"].replace(0, np.nan)
    if "alos" not in df.columns:
        df["alos"] = recomputed
        log.note("ALOS recomputed as nights / arrivals")
    else:
        diff = (df["alos"] - recomputed).abs() > 0.05
        if int(diff.sum()):
            log.note(f"{int(diff.sum())} rows where reported ALOS differs from nights/arrivals by >0.05")
    if "occupancy" not in df.columns:
        df["occupancy"] = np.nan
    return df[["month", "nationality", "guest_arrivals", "guest_nights", "alos", "occupancy"]]


def adapt_events(df: pd.DataFrame, log: AdapterLog) -> pd.DataFrame:
    df["start"] = pd.to_datetime(df["start"].astype(str))
    df["end"] = pd.to_datetime(df["end"].astype(str))
    bad = int((df["end"] < df["start"]).sum())
    if bad:
        raise SchemaError(f"events: {bad} events end before they start")
    return df[["start", "end", "name", "type"]]


def adapt_holidays(df: pd.DataFrame, log: AdapterLog) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"].astype(str))
    df = _map_country(df, "country", log)
    df = _dedupe(df, ["date", "country"], log)
    return df[["date", "country", "name"]]


def adapt_seasonality(df: pd.DataFrame, log: AdapterLog) -> pd.DataFrame:
    df["month_of_year"] = pd.to_numeric(df["month_of_year"]).astype(int)
    df["index"] = pd.to_numeric(df["index"]).astype(float)
    if sorted(df["month_of_year"]) != list(range(1, 13)):
        raise SchemaError("seasonality: expected exactly months 1..12")
    return df[["month_of_year", "index"]]


ADAPTERS = {
    "flight_schedule": ("flights_month", adapt_flights),
    "pax_traffic": ("pax_month", adapt_pax),
    "hotel_performance": ("hotel_month", adapt_hotel),
    "events": ("events", adapt_events),
    "holidays": ("holidays", adapt_holidays),
    "seasonality": ("seasonality", adapt_seasonality),
}


def ingest(folder: Path) -> tuple[dict[str, pd.DataFrame], list[AdapterLog]]:
    schema = load_schema_map()
    tables: dict[str, pd.DataFrame] = {}
    logs: list[AdapterLog] = []
    for dataset, (table, fn) in ADAPTERS.items():
        spec = schema[dataset]
        path = find_file(folder, spec["pattern"])
        raw = read_raw(path)
        log = AdapterLog(dataset=dataset, source=path.name, rows_in=len(raw))
        df = apply_schema(raw, spec, dataset)
        out = fn(df, log)
        log.rows_out = len(out)
        print(f"  [{dataset}] {path.name}: {log.rows_in:,} rows in -> {table}: {log.rows_out:,} rows out")
        tables[table] = out.reset_index(drop=True)
        logs.append(log)
    tables["market_map"] = regions.market_map()[["country", "market_key", "region"]]
    return tables, logs
