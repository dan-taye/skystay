import json

import numpy as np
import pytest

from skystay import paths
from skystay.scenario.levers import Frequency, Scenario
from skystay.sensitivity.fast import FACTORS, FastEval


@pytest.fixture(scope="module")
def fe(data_ready):
    from skystay import train
    from skystay.scenario.engine import Engine

    if not (paths.MODEL_DIR / train.BUNDLE).exists():
        train.run(bootstrap=20)
    return FastEval(Engine(train.load(), draws=100))


def test_fast_eval_matches_engine(fe):
    e = fe.e
    np.testing.assert_allclose(fe.evaluate(fe.neutral())["nights"][0], e.baseline.arrays["nights"], rtol=1e-10)
    x = fe.neutral()
    x[0] = 1.1
    r = e.run(Scenario(levers=[Frequency(pct=10)], draws=100))
    np.testing.assert_allclose(fe.evaluate(x)["guests"][0], r.arrays["guests"], rtol=1e-10)


def test_fast_eval_monotone_in_capacity(fe):
    X = np.tile(fe.neutral(), (5, 1))
    X[:, 0] = [0.85, 0.95, 1.0, 1.05, 1.15]
    tot = fe.evaluate(X)["nights"].sum(axis=(1, 2))
    assert np.all(np.diff(tot) > 0)


def test_sensitivity_report_is_complete():
    f = paths.REPORTS / "sensitivity.json"
    if not f.exists():
        pytest.skip("run `skystay sensitivity` first")
    s = json.loads(f.read_text(encoding="utf-8"))
    assert len(s["sobol"]["factors"]) == len(FACTORS)
    for r in s["sobol"]["factors"]:
        assert -0.05 <= r["S1"] <= 1.05 and -0.05 <= r["ST"] <= 1.05
    assert len(s["vulnerability"]["rows"]) == 4
    assert all(r["delta_nights"] < 0 for r in s["vulnerability"]["rows"])
    assert any(r["status"] != "served" for r in s["per_1000_seats"]["rows"])
    assert set(s["planner_brief"]["sections"]) >= {"best_routes", "seasonal_risk", "do_differently", "confidence"}
