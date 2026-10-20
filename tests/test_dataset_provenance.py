from datetime import datetime, timezone

import pytest

from app.dataset_provenance import build_dataset_provenance, verify_dataset_provenance


def _provenance(**overrides):
    values = {
        "source_name": "operator-supplied-fixture",
        "source_version": "snapshot-2026-01",
        "snapshot_sha256": "a" * 64,
        "retrieved_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "timestamp_timezone": "UTC",
        "duplicate_policy": "reject",
        "adjustment_policy": "none",
        "license_or_terms_reference": "operator-verified terms reference",
        "instruments": ["EUR/USD", "GBP/USD"],
        "timeframe": "M5",
        "row_count": 100,
    }
    values.update(overrides)
    return build_dataset_provenance(**values)


def test_provenance_normalizes_instruments_and_utc_timestamp():
    record = _provenance(instruments=["GBP/USD", "EUR/USD", "EUR/USD"])
    payload = record.to_dict()
    assert payload["instruments"] == ["EUR/USD", "GBP/USD"]
    assert payload["retrieved_at"] == "2026-01-01T00:00:00Z"


@pytest.mark.parametrize("overrides, message", [
    ({"source_name": ""}, "source_name"),
    ({"source_version": ""}, "source_version"),
    ({"snapshot_sha256": "not-a-hash"}, "snapshot_sha256"),
    ({"timestamp_timezone": ""}, "timestamp_timezone"),
    ({"duplicate_policy": ""}, "duplicate_policy"),
    ({"adjustment_policy": ""}, "adjustment_policy"),
    ({"license_or_terms_reference": ""}, "license_or_terms_reference"),
    ({"instruments": []}, "instruments"),
    ({"row_count": 0}, "row_count"),
    ({"timeframe": ""}, "timeframe"),
])
def test_required_metadata_fails_closed(overrides, message):
    with pytest.raises(ValueError, match=message):
        _provenance(**overrides)


def test_naive_retrieval_timestamp_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        _provenance(retrieved_at=datetime(2026, 1, 1))


def test_snapshot_and_parsed_facts_must_match():
    record = _provenance()
    valid = verify_dataset_provenance(
        record,
        actual_snapshot_sha256="a" * 64,
        actual_row_count=100,
        actual_instruments=["EUR/USD", "GBP/USD"],
        actual_timeframe="M5",
    )
    assert valid["verified"] is True
    assert valid["actionable"] is False
    assert valid["source_authenticated"] is False
    assert len(valid["provenance_sha256"]) == 64

    with pytest.raises(ValueError, match="SHA-256"):
        verify_dataset_provenance(record, actual_snapshot_sha256="b" * 64,
                                  actual_row_count=100, actual_instruments=["EUR/USD", "GBP/USD"],
                                  actual_timeframe="M5")
    with pytest.raises(ValueError, match="row count"):
        verify_dataset_provenance(record, actual_snapshot_sha256="a" * 64,
                                  actual_row_count=99, actual_instruments=["EUR/USD", "GBP/USD"],
                                  actual_timeframe="M5")
    with pytest.raises(ValueError, match="instruments"):
        verify_dataset_provenance(record, actual_snapshot_sha256="a" * 64,
                                  actual_row_count=100, actual_instruments=["EUR/USD"],
                                  actual_timeframe="M5")
    with pytest.raises(ValueError, match="timeframe"):
        verify_dataset_provenance(record, actual_snapshot_sha256="a" * 64,
                                  actual_row_count=100, actual_instruments=["EUR/USD", "GBP/USD"],
                                  actual_timeframe="H1")


def test_provenance_mapping_rejects_unknown_schema():
    payload = _provenance().to_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        verify_dataset_provenance(payload, actual_snapshot_sha256="a" * 64,
                                  actual_row_count=100, actual_instruments=["EUR/USD", "GBP/USD"],
                                  actual_timeframe="M5")
