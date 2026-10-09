"""Parameter recovery: on synthetic data the pipeline recovers the known ground truth.

Tolerances are stated in skystay.train.recovery and reported in reports/recovery.json.
This is evidence the METHOD works; it is not evidence of real-world accuracy.
"""

from types import SimpleNamespace

import pytest

from skystay import settings
from skystay.chain import panel
from skystay.chain import parametric as par
from skystay.train import recovery


@pytest.fixture(scope="module")
def fitted(data_ready):
    if not settings.is_synthetic():
        pytest.skip("parameter recovery needs synthetic ground truth")
    p = panel.build()
    l2 = par.fit(p, "L2")
    return SimpleNamespace(chain=l2, l2=l2, panels=p)


@pytest.mark.parametrize(
    "check", ["load_factor", "transfer_share", "alos", "non_air_share", "bridge_diagonal", "capture_rate",
              "domestic_is_non_air"]
)
def test_recovers_ground_truth(fitted, check):
    res = recovery(fitted)["checks"][check]
    assert res["pass"], f"{check}: {res['statistic']} {res['value']:.4f} > tolerance {res['tolerance']}"
