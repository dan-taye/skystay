import hashlib

import pandas as pd
import pytest

from skystay import paths, settings
from skystay.io import adapters, store, synthetic

CANONICAL = {
    "flights_month": ["month", "origin_airport", "origin_country", "airline", "aircraft_type", "flights", "seats",
                      "weekly_freq_avg", "transfer_share"],
    "pax_month": ["month", "origin_airport", "origin_country", "pax_arriving", "pax_transfer"],
    "hotel_month": ["month", "nationality", "guest_arrivals", "guest_nights", "alos", "occupancy"],
    "events": ["start", "end", "name", "type"],
    "holidays": ["date", "country", "name"],
    "seasonality": ["month_of_year", "index"],
    "market_map": ["country", "market_key", "region"],
}


@pytest.mark.parametrize("table", list(CANONICAL))
def test_canonical_schema(data_ready, table):
    df = store.read(table)
    assert list(df.columns) == CANONICAL[table]
    assert len(df) > 0


def test_types_non_negative_and_unique_keys(data_ready):
    fl = store.read("flights_month")
    px = store.read("pax_month")
    ho = store.read("hotel_month")
    assert pd.api.types.is_datetime64_any_dtype(fl["month"])
    assert (fl["month"].dt.day == 1).all(), "months must be first-of-month dates"
    for df, cols in ((fl, ["flights", "seats"]), (px, ["pax_arriving", "pax_transfer"]),
                     (ho, ["guest_arrivals", "guest_nights"])):
        assert (df[cols] >= 0).all().all()
    assert not fl.duplicated(["month", "origin_airport", "origin_country", "airline", "aircraft_type"]).any()
    assert not px.duplicated(["month", "origin_airport"]).any()
    assert not ho.duplicated(["month", "nationality"]).any()


def test_market_keys_resolve_to_regions(data_ready):
    mm = store.read("market_map").set_index("market_key")["region"]
    for col, table in (("origin_country", "flights_month"), ("nationality", "hotel_month")):
        keys = set(store.read(table)[col])
        assert keys <= set(mm.index), keys - set(mm.index)


def test_quality_flags_defects(quality_report):
    """The injected synthetic defects are detected and logged, never silently dropped."""
    if settings.data_mode() != "synthetic":
        pytest.skip("defects are injected only in synthetic mode")
    missing = {m["nationality"]: m["missing"] for m in quality_report["missing_hotel_months"]}
    assert missing.get("KW") == ["2020-08"]
    assert missing.get("TH") == ["2022-02"]
    assert missing.get("BR") == ["2023-11"]
    assert {(m["origin_airport"], m["month"]) for m in quality_report["missing_pax_months"]} == {("CMB", "2021-03")}
    hotel_notes = next(a for a in quality_report["adapters"] if a["dataset"] == "hotel_performance")["notes"]
    assert any("duplicate" in n for n in hotel_notes)
    assert quality_report["load_factor"]["clipped_at_1_05"] >= 1
    assert quality_report["overlap_window"]["n_months"] >= 60


def test_adapter_reports_unmapped_columns():
    spec = adapters.load_schema_map()["hotel_performance"]
    raw = pd.DataFrame({"Month": ["2024-01"], "Nationality": ["UK"], "Arrivals": [10], "Guest Nights": [30]})
    with pytest.raises(adapters.SchemaError) as e:
        adapters.apply_schema(raw, spec, "hotel_performance")
    assert "guest_arrivals" in str(e.value) and "Arrivals" in str(e.value)


def test_unknown_country_label_raises():
    log = adapters.AdapterLog("hotel_performance", "x.csv")
    df = pd.DataFrame({"month": ["2024-01"], "nationality": ["Atlantis"], "guest_arrivals": [1], "guest_nights": [2]})
    with pytest.raises(adapters.SchemaError, match="Atlantis"):
        adapters.adapt_hotel(df, log)


def test_country_aliases_map():
    from skystay.reconcile import regions

    assert regions.to_key("UK") == "GB"
    assert regions.to_key("Korea, Republic of") == "KR"
    assert regions.to_key("Others") == "OTHER"
    assert regions.to_key("Türkiye") == "TR"


def test_generator_is_deterministic(tmp_path, data_ready):
    if settings.data_mode() != "synthetic":
        pytest.skip("synthetic only")
    synthetic.generate(out_dir=tmp_path)

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    for name in ("hotel_performance.csv", "pax_traffic.csv", "ground_truth.json"):
        assert digest(tmp_path / name) == digest(paths.SYNTH_GEN / name), name
