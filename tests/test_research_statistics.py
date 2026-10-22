import pytest

from app.research_statistics import target_first_rate_summary


def test_target_first_rate_uses_scipy_jeffreys_interval():
    result = target_first_rate_summary(target_first=7, stop_first=3)
    assert result["resolved_barrier_outcomes"] == 10
    assert result["target_first_share"] == 0.7
    assert 0 < result["credible_interval"][0] < 0.7 < result["credible_interval"][1] < 1
    assert result["method"] == "Jeffreys beta-binomial credible interval"


def test_empty_sample_does_not_report_a_rate():
    result = target_first_rate_summary(target_first=0, stop_first=0)
    assert result["target_first_share"] is None
    assert result["credible_interval"] is None


@pytest.mark.parametrize("target,stop", [(-1, 0), (0, -1), (True, 0), (1.5, 0)])
def test_rejects_invalid_counts(target, stop):
    with pytest.raises(ValueError):
        target_first_rate_summary(target_first=target, stop_first=stop)


def test_rejects_invalid_credible_level():
    with pytest.raises(ValueError):
        target_first_rate_summary(target_first=1, stop_first=1, credible_level=1.0)
