"""Repository paths. Everything is resolved relative to the repo root."""

from __future__ import annotations

import os
from pathlib import Path


def _find_root() -> Path:
    env = os.environ.get("SKYSTAY_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "config").exists():
            return parent
    return here.parents[2]


ROOT = _find_root()
CONFIG = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
SYNTH = DATA / "synthetic"
SYNTH_GEN = SYNTH / "generated"
PROCESSED = DATA / "processed"
MODEL_DIR = PROCESSED / "model"
DB_PATH = PROCESSED / "skystay.duckdb"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DOCS = ROOT / "docs"
VIDEO = ROOT / "video"
VIDEO_OUT = VIDEO / "out"
DECK = ROOT / "deck"
DECK_OUT = DECK / "out"
DEMO = ROOT / "demo"

SETTINGS_FILE = CONFIG / "settings.yaml"
ASSUMPTIONS_FILE = CONFIG / "assumptions.yaml"
REGIONS_FILE = CONFIG / "regions.yaml"
SCHEMA_MAP_FILE = CONFIG / "schema_map.yaml"
TEAM_FILE = ROOT / "team.yaml"


def ensure_dirs() -> None:
    for d in (PROCESSED, MODEL_DIR, REPORTS, FIGURES, VIDEO_OUT, DECK_OUT, DEMO):
        d.mkdir(parents=True, exist_ok=True)
