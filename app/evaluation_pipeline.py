"""Orchestration for reproducible, offline research evaluation.

This prepares and records a validated research run. It does not train models, create
predictions, authorize trading, or alter the serving API. Scoring is only permitted
after callers provide explicitly configured outcomes and aligned baselines.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from app.block_bootstrap import BlockBootstrapComparison, compare_with_block_bootstrap
from app.cost_aware_evaluation import CostAwareSummary, RoundTripCosts, score_outcomes
from app.dataset_provenance import DatasetProvenance
from app.evaluation import EvaluationConfig, ForexBar, validate_bars
from app.outcome_labeling import OutcomeLabelResult
from app.run_manifest import ExperimentRegistry, RunManifest, build_run_manifest
from app.walk_forward import WalkForwardPlan, generate_walk_forward_folds


@dataclass(frozen=True)
class PreparedResearchRun:
    manifest: RunManifest
    quality_report: dict[str, Any]
    fold_plans: tuple[WalkForwardPlan, ...]
    registry_inserted: bool


@dataclass(frozen=True)
class ScoredFoldComparison:
    candidate_summary: CostAwareSummary
    baseline_summary: CostAwareSummary
    uncertainty: BlockBootstrapComparison
    excluded_candidate: int
    excluded_baseline: int


def prepare_research_run(
    bars: Sequence[ForexBar],
    *,
    config: EvaluationConfig,
    code_commit: str,
    model_id: str,
    model_revision: str,
    initial_train_bars: int,
    validation_bars: int,
    test_bars: int,
    step_bars: int,
    registry: ExperimentRegistry,
    dataset_provenance: DatasetProvenance | Mapping[str, Any] | None = None,
    actual_snapshot_sha256: str | None = None,
) -> PreparedResearchRun:
    """Validate one explicitly configured dataset and register per-symbol fold plans.

    Input rows must already be chronological per symbol/timeframe. The function refuses
    mixed timeframes, unrequested symbols, duplicate/non-monotonic bars, invalid OHLC,
    and missing timestamps. It deliberately does not sort, fill, fetch, or silently drop.

    For provenance-bound runs, callers must provide both a provenance record and the
    SHA-256 digest of the exact original snapshot bytes before parsing/transformation.
    The digest is consistency evidence, not source authentication.
    """
    config.validate()
    if (dataset_provenance is None) != (actual_snapshot_sha256 is None):
        raise ValueError(
            "dataset_provenance and actual_snapshot_sha256 must be supplied together"
        )
    if not bars:
        raise ValueError("bars must not be empty")
    quality = validate_bars(bars)
    if quality.invalid_rows or quality.duplicate_timestamps or quality.non_monotonic_timestamps:
        raise ValueError("dataset failed quality checks; quarantine/fix rows before evaluation")
    if set(quality.symbols) != set(config.symbols):
        raise ValueError("dataset symbols must exactly match the explicit evaluation config")
    if quality.timeframes != (config.timeframe,):
        raise ValueError("dataset timeframe must exactly match the evaluation config")

    plans: list[WalkForwardPlan] = []
    for symbol in config.symbols:
        symbol_bars = [bar for bar in bars if bar.symbol == symbol]
        plan = generate_walk_forward_folds(
            symbol_bars,
            initial_train_bars=initial_train_bars,
            validation_bars=validation_bars,
            test_bars=test_bars,
            step_bars=step_bars,
            horizon_bars=config.horizon_bars,
        )
        plans.append(plan)

    manifest = build_run_manifest(
        bars,
        code_commit=code_commit,
        model_id=model_id,
        model_revision=model_revision,
        dataset_id=config.dataset_version,
        evaluation_config=asdict(config),
        dataset_provenance=dataset_provenance,
        actual_snapshot_sha256=actual_snapshot_sha256,
    )
    inserted = registry.register(manifest)
    return PreparedResearchRun(
        manifest=manifest,
        quality_report=asdict(quality),
        fold_plans=tuple(plans),
        registry_inserted=inserted,
    )


def compare_scored_outcomes(
    candidate_outcomes: Sequence[OutcomeLabelResult],
    baseline_outcomes: Sequence[OutcomeLabelResult],
    *,
    costs: RoundTripCosts,
    block_length: int,
    bootstrap_resamples: int = 5000,
    confidence_level: float = 0.95,
    random_seed: int = 20261018,
) -> ScoredFoldComparison:
    """Cost-score candidate and baseline, then compare aligned net outcomes.

    Only rows with a scored gross price change are retained. Because the statistical
    comparison is paired, the retained candidate/baseline observations must have matching
    identities and chronological order; mismatches fail closed. Costs are subtracted
    identically from both methods. A positive interval is not a profitability claim.
    """
    candidate_scored, candidate_summary = score_outcomes(candidate_outcomes, costs=costs)
    baseline_scored, baseline_summary = score_outcomes(baseline_outcomes, costs=costs)

    def identity(item: Any) -> tuple[str, str, str]:
        return (item.symbol, item.timeframe, item.entry_timestamp)

    candidate_ids = [identity(item) for item in candidate_scored]
    baseline_ids = [identity(item) for item in baseline_scored]
    if candidate_ids != baseline_ids:
        raise ValueError("candidate and baseline scored outcomes must be paired and chronologically aligned")
    if not candidate_scored:
        raise ValueError("no paired scored outcomes remain after exclusions")

    uncertainty = compare_with_block_bootstrap(
        [item.net_price_change for item in candidate_scored],
        [item.net_price_change for item in baseline_scored],
        block_length=block_length,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    return ScoredFoldComparison(
        candidate_summary=candidate_summary,
        baseline_summary=baseline_summary,
        uncertainty=uncertainty,
        excluded_candidate=len(candidate_outcomes) - len(candidate_scored),
        excluded_baseline=len(baseline_outcomes) - len(baseline_scored),
    )
