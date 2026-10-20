"""Descriptive fold uncertainty and point-in-time regime validation for offline research."""
from __future__ import annotations
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Mapping, Sequence

_METRICS = ("brier_score", "log_loss", "accuracy_at_fixed_half", "expected_calibration_error")


def _finite(test: Any, key: str) -> float | None:
    if not isinstance(test, Mapping):
        return None
    value = test.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        return None
    return float(value)


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp")
    value = value.strip()
    if not (value.endswith("Z") or value.endswith("+00:00")):
        raise ValueError(f"{field} must explicitly use UTC (Z or +00:00)")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must explicitly use UTC (Z or +00:00)")
    return parsed.astimezone(timezone.utc)


def _boundaries(value: Any) -> dict[int, tuple[datetime, datetime]]:
    if value is None:
        return {}
    if not isinstance(value, list):
        raise ValueError("fold_temporal_boundaries must be a list of fold boundary objects")
    output: dict[int, tuple[datetime, datetime]] = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("each fold temporal boundary must be an object")
        index = row.get("fold_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in output:
            raise ValueError("fold temporal boundaries require unique non-negative integer indices")
        start = _parse_utc(row.get("test_start_utc"), field=f"fold {index} test_start_utc")
        end = _parse_utc(row.get("test_end_utc"), field=f"fold {index} test_end_utc")
        if start > end:
            raise ValueError(f"fold {index} test_start_utc must not be after test_end_utc")
        output[index] = (start, end)
    return output


def _parse_annotations(annotations: Mapping[str, Any] | None, boundaries: Mapping[int, tuple[datetime, datetime]]) -> tuple[dict[int, str], dict[str, Any]]:
    base = {"source": None, "method": None, "source_snapshot_sha256": None, "folds": {},
            "temporal_validity_checked": False, "withheld_fold_indices": []}
    if annotations is None:
        return {}, {"status": "NO_REGIME_ANNOTATIONS", **base}
    if not isinstance(annotations, Mapping):
        raise ValueError("fold_regime_annotations must be a JSON object")
    source, method, raw = annotations.get("source"), annotations.get("method"), annotations.get("folds")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("regime annotation source must be explicitly declared")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("regime annotation method must be explicitly declared")
    if not isinstance(raw, Mapping):
        raise ValueError("regime annotation folds must be an object")
    # Legacy labels remain readable but cannot establish point-in-time validity.
    if raw and all(isinstance(v, str) for v in raw.values()):
        normalized = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.isdigit() or not value.strip():
                raise ValueError("legacy regime labels require non-negative fold keys and non-empty labels")
            normalized[str(int(key))] = value.strip()
        return {}, {"status": "LEGACY_LABELS_WITHOUT_TEMPORAL_PROOF", "source": source.strip(),
                    "method": method.strip(), "source_snapshot_sha256": None, "folds": normalized,
                    "temporal_validity_checked": False, "withheld_fold_indices": sorted(int(k) for k in normalized)}
    snapshot = annotations.get("source_snapshot_sha256")
    if not isinstance(snapshot, str) or len(snapshot) != 64:
        raise ValueError("regime annotation source_snapshot_sha256 must be a SHA-256 digest")
    try:
        bytes.fromhex(snapshot)
    except ValueError as exc:
        raise ValueError("regime annotation source_snapshot_sha256 must be hexadecimal") from exc
    if not boundaries:
        return {}, {"status": "WITHHELD_MISSING_FOLD_TEMPORAL_BOUNDARIES", "source": source.strip(),
                    "method": method.strip(), "source_snapshot_sha256": snapshot.lower(), "folds": {},
                    "temporal_validity_checked": False,
                    "withheld_fold_indices": sorted(int(k) for k in raw if isinstance(k, str) and k.isdigit())}
    labels: dict[int, str] = {}
    normalized: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.isdigit():
            raise ValueError("regime annotation fold keys must be non-negative integer strings")
        index = int(key)
        if index not in boundaries:
            raise ValueError(f"regime annotation references fold {index} without temporal boundaries")
        if not isinstance(value, Mapping):
            raise ValueError("new regime annotations must be objects with label and temporal validity fields")
        label = value.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"fold {index} regime label must be a non-empty string")
        available = _parse_utc(value.get("available_at_utc"), field=f"fold {index} available_at_utc")
        valid_from = _parse_utc(value.get("valid_from_utc"), field=f"fold {index} valid_from_utc")
        valid_through = _parse_utc(value.get("valid_through_utc"), field=f"fold {index} valid_through_utc")
        test_start, test_end = boundaries[index]
        if available > test_start:
            raise ValueError(f"fold {index} regime annotation was not available by test start")
        if valid_from > test_start or valid_through < test_end:
            raise ValueError(f"fold {index} regime annotation does not cover the entire test window")
        if valid_from > valid_through:
            raise ValueError(f"fold {index} valid_from_utc must not be after valid_through_utc")
        labels[index] = label.strip()
        normalized[str(index)] = {"label": label.strip(),
            "available_at_utc": available.isoformat().replace("+00:00", "Z"),
            "valid_from_utc": valid_from.isoformat().replace("+00:00", "Z"),
            "valid_through_utc": valid_through.isoformat().replace("+00:00", "Z")}
    return labels, {"status": "DECLARED_AND_TEMPORALLY_VALIDATED", "source": source.strip(),
                    "method": method.strip(), "source_snapshot_sha256": snapshot.lower(), "folds": normalized,
                    "temporal_validity_checked": True, "withheld_fold_indices": []}


def build_regime_uncertainty_audit(fold_results: Mapping[str, Sequence[Mapping[str, Any]]],
        paired_comparisons: Sequence[Mapping[str, Any]], fold_regime_annotations: Mapping[str, Any] | None = None,
        fold_temporal_boundaries: Any = None, *, minimum_folds_per_regime: int = 2) -> dict[str, Any]:
    """Describe fold dispersion and stratify only declared regimes valid at test time."""
    if isinstance(minimum_folds_per_regime, bool) or not isinstance(minimum_folds_per_regime, int) or minimum_folds_per_regime < 2:
        raise ValueError("minimum_folds_per_regime must be an integer of at least 2")
    boundaries = _boundaries(fold_temporal_boundaries)
    baselines = {r.get("baseline_name") for r in paired_comparisons
                 if isinstance(r, Mapping) and isinstance(r.get("baseline_name"), str)}
    candidates = sorted(name for name in fold_results if name not in baselines)
    indexed = {name: {r["fold_index"]: r for r in fold_results[name]
        if isinstance(r, Mapping) and isinstance(r.get("fold_index"), int)
        and not isinstance(r.get("fold_index"), bool) and r["fold_index"] >= 0} for name in candidates}
    common = set.intersection(*(set(indexed[name]) for name in candidates)) if candidates else set()
    complete = [i for i in sorted(common) if all(
        _finite(indexed[name][i].get("test"), "brier_score") is not None
        and _finite(indexed[name][i].get("test"), "log_loss") is not None for name in candidates)]
    labels, annotation_report = _parse_annotations(fold_regime_annotations, boundaries)
    envelopes: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    for name in candidates:
        folds = []
        for i in complete:
            test = indexed[name][i].get("test")
            folds.append({"fold_index": i, "sample_count": test.get("sample_count") if isinstance(test, Mapping) else None,
                          **{metric: _finite(test, metric) for metric in _METRICS}})
        metrics = {}
        for metric in _METRICS:
            values = [row[metric] for row in folds if row[metric] is not None]
            if values:
                metrics[metric] = {"fold_count": len(values), "minimum": min(values),
                    "q1": _quantile(values, .25), "median": _quantile(values, .5), "q3": _quantile(values, .75),
                    "maximum": max(values), "range": max(values) - min(values)}
        envelopes.append({"candidate_name": name, "fold_count": len(folds), "folds": folds,
                          "metrics": metrics, "actionable": False})
        by_regime: dict[str, list[dict[str, Any]]] = {}
        for row in folds:
            if labels.get(row["fold_index"]):
                by_regime.setdefault(labels[row["fold_index"]], []).append(row)
        for label in sorted(by_regime):
            rows = by_regime[label]
            if len(rows) < minimum_folds_per_regime:
                continue
            summary = {}
            for metric in _METRICS:
                values = [row[metric] for row in rows if row[metric] is not None]
                if values:
                    summary[metric] = {"fold_count": len(values), "median": _quantile(values, .5),
                        "minimum": min(values), "maximum": max(values), "range": max(values) - min(values)}
            strata.append({"candidate_name": name, "regime": label,
                "fold_indices": [row["fold_index"] for row in rows], "fold_count": len(rows),
                "metrics": summary, "actionable": False})
    annotation_report["minimum_folds_per_regime"] = minimum_folds_per_regime
    annotation_report["stratified_results"] = strata
    return {"protocol": "PER_FOLD_DISPERSION_AND_POINT_IN_TIME_REGIME_STRATIFICATION",
        "candidate_count": len(candidates), "common_complete_test_fold_indices": complete,
        "uncertainty_envelopes": envelopes, "regime_annotations": annotation_report,
        "confidence_intervals_produced": False, "model_selection_performed": False,
        "test_metrics_used_for_selection": False, "actionable": False,
        "interpretation": ("Fold quartiles and ranges are descriptive, not confidence intervals. Regime stratification "
            "requires explicit availability and validity timestamps covering the actual test window plus a declared "
            "source snapshot SHA-256. These checks validate internal temporal consistency, not source truth or causal "
            "correctness. Legacy string labels remain visible but are withheld from stratification. Two folds is not "
            "evidence of statistical power. Overlapping test windows may reuse observations. No pooling, significance "
            "testing, model selection, P&L, profitability claim, or execution authorization is produced.")}
}
