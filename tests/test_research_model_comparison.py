import pytest
from app.research_model_comparison import compare_model_outcomes


def test_small_samples_are_descriptive_only():
    result = compare_model_outcomes({"model-a": {"TARGET_FIRST": 12, "STOP_FIRST": 8}, "model-b": {"TARGET_FIRST": 10, "STOP_FIRST": 10}}, minimum_resolved_samples=30)
    assert result["comparisons"] == []
    assert result["models"]["model-a"]["target_first_share"] == 0.6
    assert result["models"]["model-a"]["eligible_for_inference"] is False


def test_pairwise_comparison_uses_holm_adjustment():
    result = compare_model_outcomes({"model-a": {"TARGET_FIRST": 45, "STOP_FIRST": 5}, "model-b": {"TARGET_FIRST": 25, "STOP_FIRST": 25}, "model-c": {"TARGET_FIRST": 20, "STOP_FIRST": 30}}, minimum_resolved_samples=30)
    assert len(result["comparisons"]) == 3
    assert all(0 <= row["p_value"] <= 1 for row in result["comparisons"])
    assert all(row["p_value"] <= row["p_value_holm_adjusted"] <= 1 for row in result["comparisons"])
    assert result["accuracy_metrics_published"] is False


def test_zero_sample_has_no_share_or_interval():
    result = compare_model_outcomes({"empty": {}}, minimum_resolved_samples=10)
    assert result["models"]["empty"]["target_first_share"] is None
    assert result["models"]["empty"]["credible_interval"] is None


@pytest.mark.parametrize("counts", [{"TARGET_FIRST": -1, "STOP_FIRST": 0}, {"TARGET_FIRST": True, "STOP_FIRST": 0}, {"TARGET_FIRST": 1.5, "STOP_FIRST": 0}])
def test_invalid_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        compare_model_outcomes({"model": counts})


def test_invalid_minimum_sample_size_is_rejected():
    with pytest.raises(ValueError):
        compare_model_outcomes({}, minimum_resolved_samples=0)
