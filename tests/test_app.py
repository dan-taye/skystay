"""AppTest smoke tests: every page loads with no exceptions; presets load deterministic scenarios."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from skystay import paths, settings  # noqa: E402

PAGES = ["overview", "how", "simulator", "chain", "sensitivity", "validation", "assumptions", "brief"]


@pytest.fixture(scope="module", autouse=True)
def artefacts(data_ready):
    from skystay import train

    missing = [f for f in ("metrics.json", "sensitivity.json") if not (paths.REPORTS / f).exists()]
    if missing or not (paths.MODEL_DIR / train.BUNDLE).exists():
        pytest.skip(f"run `skystay all` first (missing: {missing or 'model bundle'})")


def _app(page: str) -> AppTest:
    code = f"from skystay.app.views import {page}\n{page}.render()\n"
    return AppTest.from_string(code, default_timeout=120)


@pytest.mark.parametrize("page", PAGES)
def test_page_loads(page):
    at = _app(page).run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.title, f"{page}: no title rendered"
    if settings.is_synthetic():
        html = " ".join(m.value for m in at.markdown)
        assert "SYNTHETIC DEMO DATA" in html, f"{page}: synthetic badge missing"


@pytest.mark.parametrize("preset,key,value", [("new_route", "nr_on", True), ("winter_cut", "cap_pct", -30)])
def test_presets(preset, key, value):
    at = _app("simulator")
    at.query_params["preset"] = preset
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state[key] == value
    html = " ".join(m.value for m in at.markdown)
    assert "Change in hotel guests" in html


def test_simulator_lever_interaction():
    at = _app("simulator").run()
    at.slider(key="cap_pct").set_value(20).run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "+20% frequency" in html
