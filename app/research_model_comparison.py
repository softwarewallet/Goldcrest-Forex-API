"""Conservative pairwise comparisons of research barrier outcomes."""
from __future__ import annotations
from math import isfinite
from typing import Mapping
from scipy.stats import beta, fisher_exact


def compare_model_outcomes(counts_by_model: Mapping[str, Mapping[str, int]], *, minimum_resolved_samples: int = 30, credible_level: float = 0.95) -> dict:
    """Compare target-first shares; inference is withheld below the sample floor."""
    if isinstance(minimum_resolved_samples, bool) or not isinstance(minimum_resolved_samples, int) or minimum_resolved_samples < 1:
        raise ValueError("minimum_resolved_samples must be a positive integer")
    if isinstance(credible_level, bool) or not isinstance(credible_level, (int, float)) or not isfinite(credible_level) or not 0 < credible_level < 1:
        raise ValueError("credible_level must be strictly between 0 and 1")
    summaries = {}
    for model_id, counts in sorted(counts_by_model.items()):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model identifiers must be non-empty strings")
        if not isinstance(counts, Mapping):
            raise ValueError("each model must provide a mapping of outcome counts")
        target, stop = counts.get("TARGET_FIRST", 0), counts.get("STOP_FIRST", 0)
        for value in (target, stop):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("outcome counts must be non-negative integers")
        n = target + stop
        if n:
            alpha = (1 - credible_level) / 2
            interval = [float(beta.ppf(alpha, target + 0.5, stop + 0.5)), float(beta.ppf(1 - alpha, target + 0.5, stop + 0.5))]
            share = target / n
        else:
            share, interval = None, None
        summaries[model_id] = {"target_first": target, "stop_first": stop, "resolved_barrier_outcomes": n, "target_first_share": share, "credible_interval": interval, "eligible_for_inference": n >= minimum_resolved_samples}
    eligible = [key for key, value in summaries.items() if value["eligible_for_inference"]]
    comparisons = []
    for i, left in enumerate(eligible):
        for right in eligible[i + 1:]:
            a, b = summaries[left], summaries[right]
            odds_ratio, p_value = fisher_exact([[a["target_first"], a["stop_first"]], [b["target_first"], b["stop_first"]]], alternative="two-sided")
            comparisons.append({"model_a": left, "model_b": right, "odds_ratio": float(odds_ratio), "p_value": float(p_value), "p_value_holm_adjusted": None})
    ordered = sorted(range(len(comparisons)), key=lambda idx: comparisons[idx]["p_value"])
    running_max, total = 0.0, len(ordered)
    for rank, idx in enumerate(ordered):
        adjusted = min(1.0, comparisons[idx]["p_value"] * (total - rank))
        running_max = max(running_max, adjusted)
        comparisons[idx]["p_value_holm_adjusted"] = running_max
    return {"schema_version": "1.0", "minimum_resolved_samples": minimum_resolved_samples, "credible_level": float(credible_level), "models": summaries, "comparisons": comparisons, "method": "two-sided Fisher exact test with Holm-Bonferroni correction", "accuracy_metrics_published": False, "interpretation": "Descriptive TARGET_FIRST versus STOP_FIRST outcomes only; not prediction accuracy, calibrated probability, expected return, or profitability. Differences may reflect instrument, regime, horizon, selection, or coverage differences."}
