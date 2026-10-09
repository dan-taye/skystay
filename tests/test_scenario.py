"""Golden tests for the scenario engine."""

import time

import numpy as np
import pytest

from skystay.scenario import presets
from skystay.scenario.engine import Engine, ScenarioError
from skystay.scenario.levers import (
    Capture,
    DiscontinueRoute,
    Event,
    Frequency,
    LoadFactor,
    MixShift,
    NewRoute,
    Scenario,
    SeasonShift,
    TransferShare,
)


@pytest.fixture(scope="module")
def engine(data_ready):
    from skystay import paths, train

    if not (paths.MODEL_DIR / train.BUNDLE).exists():
        train.run(bootstrap=20)
    return Engine(train.load(), draws=300)


def _g(res):
    return res.table.set_index(["month", "market"])["guests"]


def test_zero_change_equals_baseline(engine):
    r = engine.run(Scenario(name="no change", draws=engine.draws))
    np.testing.assert_array_equal(_g(r).to_numpy(), _g(engine.baseline).to_numpy())
    np.testing.assert_array_equal(r.table["lo80"].to_numpy(), engine.baseline.table["lo80"].to_numpy())
    assert (r.table["delta_vs_baseline"] == 0).all()
    assert r.totals["delta_guests"] == 0


@pytest.mark.parametrize("market", ["GB", "IN", "SA", "CN", "US"])
def test_more_seats_never_lowers_guests(engine, market):
    base = _g(engine.baseline)
    for pct in (5, 20, 60):
        r = engine.run(Scenario(levers=[Frequency(market=market, pct=pct)], draws=engine.draws))
        g = _g(r)
        assert (g >= base - 1e-6).all(), f"{market} +{pct}%: some market lost guests"
        assert g.sum() > base.sum()


def test_removing_routes_only_affects_bridge_linked_markets(engine):
    rt = engine.route_table()
    origin = "JP"
    levers = [DiscontinueRoute(route_id=rid) for rid in rt.loc[rt["origin"] == origin, "route_id"]]
    r = engine.run(Scenario(levers=levers, draws=engine.draws))
    W = engine.baseline.arrays["W"]
    oi = engine.baseline.arrays["origins"].index(origin)
    linked = W[oi].sum(axis=1) > 0
    d = r.table.groupby("market")["delta_vs_baseline"].sum().reindex(engine.nats).to_numpy()
    assert (np.abs(d[~linked]) < 1e-6).all(), "unlinked nationality changed"
    assert d[linked].sum() < 0
    assert d[engine.nats.index("JP")] < 0


def test_deterministic_with_seed(engine):
    sc = presets.presets()["new_route"].model_copy(update={"draws": engine.draws})
    a, b = engine.run(sc), engine.run(sc)
    np.testing.assert_array_equal(a.table[["guests", "lo80", "hi95"]].to_numpy(),
                                  b.table[["guests", "lo80", "hi95"]].to_numpy())
    c = engine.run(sc.model_copy(update={"seed": 7}))
    np.testing.assert_array_equal(a.table["guests"].to_numpy(), c.table["guests"].to_numpy())
    assert not np.array_equal(a.table["lo80"].to_numpy(), c.table["lo80"].to_numpy())


def test_intervals_bracket_point_and_widen_for_cold_start(engine):
    t = engine.baseline.table
    assert (t["lo95"] <= t["lo80"]).all() and (t["hi80"] <= t["hi95"]).all()
    r = engine.run(Scenario(levers=[NewRoute(origin="ZA", weekly_freq=7)], draws=engine.draws))
    assert any("cold-start" in w for w in r.warnings)
    za = r.table[r.table["market"] == "ZA"]
    zb = engine.baseline.table[engine.baseline.table["market"] == "ZA"]
    width_new = ((za["hi80"] - za["lo80"]) / za["guests"]).mean()
    width_old = ((zb["hi80"] - zb["lo80"]) / zb["guests"]).mean()
    assert width_new > width_old
    assert za["cold_start"].all()


def test_guardrails_reject_impossible_inputs(engine):
    with pytest.raises(ScenarioError, match="100%"):
        engine.run(Scenario(levers=[LoadFactor(market="PH", delta_pp=25)]))
    with pytest.raises(ScenarioError):
        engine.run(Scenario(levers=[TransferShare(market="AU", delta_pp=40)]))
    with pytest.raises(ValueError):  # pydantic: frequency cannot fall below -100%
        Frequency(market="GB", pct=-150)
    with pytest.raises(ValueError):
        LoadFactor(market="GB", value=1.2)


def test_extrapolation_warning(engine):
    r = engine.run(Scenario(levers=[Frequency(market="GB", pct=150)], draws=engine.draws))
    assert any("extrapolation" in w for w in r.warnings)
    assert r.table.loc[r.table["market"] == "GB", "extrapolation"].all()


def test_extremes_have_no_nans(engine):
    levers = [Frequency(market="IN", pct=-100), LoadFactor(market="GB", value=1.0), Capture(market="ALL", pct=50),
              SeasonShift(market="ALL", from_season="Summer low (Jun-Aug)", to_season="Winter peak (Dec-Feb)", pct=50),
              MixShift(from_market="PK", to_market="VN", pct=40), Event(month=12, days=10)]
    r = engine.run(Scenario(levers=levers, draws=engine.draws))
    num = r.table.select_dtypes("number")
    assert np.isfinite(num.to_numpy()).all()
    assert (r.table["guests"] >= 0).all()


def test_event_lever_raises_guests(engine):
    r = engine.run(Scenario(levers=[Event(month=6, days=7)], draws=engine.draws))
    assert r.totals["delta_guests"] >= 0


def test_speed_under_300ms(engine):
    sc = presets.presets()["winter_cut"].model_copy(update={"draws": 500})
    engine.run(sc)
    t = time.perf_counter()
    for _ in range(3):
        engine.run(sc)
    assert (time.perf_counter() - t) / 3 < 0.3
