"""SciPy-backed statistical summaries for offline research telemetry only."""
from math import isfinite

from scipy.stats import beta


def target_first_rate_summary(*, target_first: int, stop_first: int, credible_level: float = 0.95) -> dict:
    """Return a Jeffreys-prior interval for target-hit share among resolved barrier hits.

    This interval describes the recorded sample only; it does not correct selection
    bias, serial dependence, transaction costs, or establish predictive skill.
    """
    for name, value in (("target_first", target_first), ("stop_first", stop_first)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(credible_level, bool) or not isinstance(credible_level, (int, float)):
        raise ValueError("credible_level must be between 0 and 1")
    if not isfinite(credible_level) or not 0 < credible_level < 1:
        raise ValueError("credible_level must be between 0 and 1")

    n = target_first + stop_first
    if n == 0:
        return {
            "method": "Jeffreys beta-binomial credible interval",
            "credible_level": float(credible_level),
            "resolved_barrier_outcomes": 0,
            "target_first_count": 0,
            "stop_first_count": 0,
            "target_first_share": None,
            "credible_interval": None,
        }

    alpha = (1.0 - float(credible_level)) / 2.0
    posterior_a = target_first + 0.5
    posterior_b = stop_first + 0.5
    lower, upper = beta.ppf([alpha, 1.0 - alpha], posterior_a, posterior_b)
    return {
        "method": "Jeffreys beta-binomial credible interval",
        "credible_level": float(credible_level),
        "resolved_barrier_outcomes": n,
        "target_first_count": target_first,
        "stop_first_count": stop_first,
        "target_first_share": target_first / n,
        "credible_interval": [float(lower), float(upper)],
    }
