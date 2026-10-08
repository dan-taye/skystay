import json

import pytest

from skystay import paths


@pytest.fixture(scope="session")
def data_ready():
    """Canonical tables exist (runs `skystay data` once if needed)."""
    if not paths.DB_PATH.exists() or not (paths.REPORTS / "data_quality.json").exists():
        from skystay.io import pipeline

        pipeline.run()
    return paths.DB_PATH


@pytest.fixture(scope="session")
def quality_report(data_ready):
    return json.loads((paths.REPORTS / "data_quality.json").read_text(encoding="utf-8"))
