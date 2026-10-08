"""Canonical DuckDB store (data/processed/skystay.duckdb)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from skystay import paths

TABLES = ["flights_month", "pax_month", "hotel_month", "events", "holidays", "seasonality", "market_map"]


def write_tables(tables: dict[str, pd.DataFrame], db_path: Path | None = None) -> None:
    db_path = db_path or paths.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        for name, df in tables.items():
            con.register("_tmp", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp")
            con.unregister("_tmp")


def read(table: str, db_path: Path | None = None) -> pd.DataFrame:
    db_path = db_path or paths.DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found: run `skystay data` first")
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(f"SELECT * FROM {table}").df()


def read_all(db_path: Path | None = None) -> dict[str, pd.DataFrame]:
    return {t: read(t, db_path) for t in TABLES}


def query(sql: str, db_path: Path | None = None) -> pd.DataFrame:
    db_path = db_path or paths.DB_PATH
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(sql).df()
