"""No leakage: everything estimated for a back-test fold uses only months before its cutoff."""

import numpy as np
import pandas as pd
import pytest

from skystay.backtest import runner
from skystay.backtest.splits import make_folds
from skystay.chain import panel
from skystay.chain import parametric as par
from skystay.ml import features


@pytest.fixture(scope="module")
def panels(data_ready):
    return panel.build()


def test_folds_are_ordered_and_disjoint(panels):
    folds = make_folds(panels.months, 3, 12, 12)
    assert [f.holdout for f in folds] == [False, False, False, True]
    for f in folds:
        assert f.test[0] == f.cutoff and len(f.test) == 12
        assert f.train_months == int((panels.months < f.cutoff).sum())
    assert folds[-1].test[-1] == panels.months[-1]
    cut = [f.cutoff for f in folds]
    assert cut == sorted(cut)


def test_restrict_excludes_cutoff(panels):
    cutoff = pd.Timestamp("2023-01-01")
    ptr = panel.restrict(panels, cutoff)
    for df in (ptr.origin_month, ptr.nat_month, ptr.routes):
        assert df["month"].max() < cutoff
    assert ptr.months.max() < cutoff


def test_fold_training_never_reaches_cutoff(panels):
    """Run a real fold and check every logged training frame (panels, ML rows, lag features)."""
    runner.TRAIN_LOG.clear()
    fold = make_folds(panels.months, 3, 12, 12)[0]
    out = runner.run_fold(panels, fold, levels=("L2",))
    assert runner.TRAIN_LOG, "no training frames were logged"
    whats = {r["what"] for r in runner.TRAIN_LOG}
    assert {"origin_month", "nat_month", "ml_train", "lag12_features"} <= whats
    for r in runner.TRAIN_LOG:
        assert r["last_month"] < r["cutoff"], r
    # the fold covers a route launch (KE, 2022-10): predictions exist and are finite
    ke = out[(out["nationality"] == "KE") & (out["model"] == "hybrid")]
    assert len(ke) == 12 and np.isfinite(ke["pred_guests"]).all()


def test_guard_raises_on_leak(panels):
    fold = make_folds(panels.months, 3, 12, 12)[0]
    with pytest.raises(runner.LeakageError):
        runner.guard("bad", fold, pd.Series([fold.cutoff]))


def test_future_data_cannot_change_a_fold_fit(panels):
    """Scaling all post-cutoff guests and traffic by 10x must not change the fold's predictions."""
    cutoff = pd.Timestamp("2024-01-01")
    test = panels.months[(panels.months >= cutoff)][:12]
    bad = panel.Panels(**{**panels.__dict__})
    bad.nat_month = panels.nat_month.copy()
    bad.origin_month = panels.origin_month.copy()
    fut = bad.nat_month["month"] >= cutoff
    bad.nat_month.loc[fut, ["guests", "nights"]] *= 10
    futo = bad.origin_month["month"] >= cutoff
    bad.origin_month.loc[futo, ["pax", "transfer", "terminating"]] *= 10

    def fit_predict(pp):
        m = par.fit(panel.restrict(pp, cutoff), "L2")
        te = features.frame(pp, m, test, use_actual_supply=False)
        return te["guests_pred"].to_numpy(), te["lag12"].to_numpy()

    a, la = fit_predict(panels)
    b, lb = fit_predict(bad)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(np.nan_to_num(la), np.nan_to_num(lb))
