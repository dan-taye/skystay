"""Market keys and regions from config/regions.yaml (single source of truth)."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd
import yaml

from skystay import paths


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


@lru_cache(maxsize=1)
def load() -> dict:
    return yaml.safe_load(paths.REGIONS_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def lookup() -> dict[str, str]:
    """Normalised name / alias / ISO2 -> ISO2."""
    out: dict[str, str] = {}
    for c in load()["countries"]:
        for key in [c["iso2"], c["name"], *c.get("aliases", [])]:
            out[_norm(key)] = c["iso2"]
    return out


def to_key(value: str) -> str | None:
    """Map a raw country/nationality label to its market key, or None if unknown."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return lookup().get(_norm(value))


def map_series(s: pd.Series) -> tuple[pd.Series, list[str]]:
    """Vectorised `to_key`; returns (keys, sorted list of unmapped raw labels)."""
    keys = s.map(to_key)
    unmapped = sorted(set(s[keys.isna()].astype(str)))
    return keys, unmapped


@lru_cache(maxsize=1)
def region_of() -> dict[str, str]:
    return {c["iso2"]: c["region"] for c in load()["countries"]}


@lru_cache(maxsize=1)
def name_of() -> dict[str, str]:
    return {c["iso2"]: c["name"] for c in load()["countries"]}


def region_names() -> dict[str, str]:
    return dict(load()["regions"])


def market_map() -> pd.DataFrame:
    """Canonical market_map table: country, market_key, region, name."""
    rows = [
        {"country": c["name"], "market_key": c["iso2"], "region": c["region"], "name": c["name"]}
        for c in load()["countries"]
    ]
    return pd.DataFrame(rows)


def label(key: str) -> str:
    return name_of().get(key, key)
