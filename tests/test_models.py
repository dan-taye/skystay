import numpy as np
import pandas as pd
import pytest

from skystay.chain import panel
from skystay.chain import parametric as par
from skystay.ml import econometric, features, gbm


@pytest.fixture(scope="module")
def p(data_ready):
    return panel.build()


@pytest.fixture(scope="module", params=["L0", "L1", "L2"])
def model(request, p):
    return par.fit(panel.restrict(p, pd.Timestamp("2025-01-01")), request.param)


def _year(p):
    return p.months[-12:]


def test_parametric_predicts_all_nationalities(p, model):
    months = _year(p)
    pred = par.predict(model, par.seats_frame(p, months), calendar=par.calendar_frame(p, months))
    assert set(pred["nationality"]) == set(p.nationalities)
    assert np.isfinite(pred[["guests_pred", "nights_pred"]]).all().all()
    assert (pred["guests_pred"] >= 0).all()
    for n in ("CA", "ZA", "AE", "OTHER"):  # no direct route
        assert pred.loc[pred["nationality"] == n, "guests_pred"].sum() > 0


def test_more_seats_never_fewer_guests(p, model):
    months = _year(p)
    seats = par.seats_frame(p, months)
    cal = par.calendar_frame(p, months)
    base = par.predict(model, seats, calendar=cal).set_index(["nationality", "month"])["guests_pred"]
    for o in ("GB", "IN", "SA"):
        more = seats.copy()
        more.loc[more["origin"] == o, "seats"] *= 1.5
        up = par.predict(model, more, calendar=cal).set_index(["nationality", "month"])["guests_pred"]
        assert (up >= base - 1e-9).all()
        assert up.sum() > base.sum()


def test_cold_start_origin_borrows_region(model):
    q = model.origin_params("VN")
    assert q["cold"]
    assert 0 < q["lf"].mean() <= 1.05
    assert q["conv"][model.nat_index()["OTHER"]].sum() > 0
    q2 = model.origin_params("CA")  # nationality exists in hotel data, no route
    assert q2["cold"] and q2["conv"][model.nat_index()["CA"]].sum() > 0


def test_econometric_and_gbm_fit(p):
    ptr = panel.restrict(p, pd.Timestamp("2025-01-01"))
    m = par.fit(ptr, "L2")
    tr = features.frame(ptr, m, ptr.months, use_actual_supply=False)
    em = econometric.fit(features.frame(ptr, m, ptr.months, use_actual_supply=True))
    s = em.summary()
    assert 0 < s["elasticity"]["coef"] < 2 and s["cov_type"] == "HC3"
    gm = gbm.fit(tr, "hybrid")
    te = features.frame(p, m, _year(p), use_actual_supply=False)
    g = gbm.predict(gm, te)
    assert np.isfinite(g).all() and (g >= 0).all()


def test_gbm_monotone_in_air_visitors(p):
    ptr = panel.restrict(p, pd.Timestamp("2025-01-01"))
    m = par.fit(ptr, "L2")
    gm = gbm.fit(features.frame(ptr, m, ptr.months, use_actual_supply=False), "hybrid")
    te = features.frame(p, m, _year(p), use_actual_supply=False)
    f0 = gbm.raw_predict(gm, te)
    te2 = te.copy()
    te2["log_air"] = te2["log_air"] + 0.3
    assert (gbm.raw_predict(gm, te2) >= f0 - 1e-12).all()
