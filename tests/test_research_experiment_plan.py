from __future__ import annotations

import copy
import hashlib

import pytest

from app.research_experiment_plan import build_experiment_plan


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sample_plan():
    return {
        "schema_version": 1,
        "plan_id": "eurusd-m15-seed-fold-sweep",
        "context": {"symbol": "EURUSD", "timeframe": "M15"},
        "config_sha256": digest(b"config-v1"),
        "dataset_sha256": digest(b"dataset-v1"),
        "code_commit_sha": "a" * 40,
        "seeds": [11, 22],
        "folds": ["fold-1", "fold-2"],
        "evaluation_windows": [
            {"window_id": "jan", "start": "2026-01-01T00:00:00Z", "end": "2026-02-01T00:00:00Z"},
            {"window_id": "feb", "start": "2026-02-01T00:00:00Z", "end": "2026-03-01T00:00:00Z"},
        ],
    }


def test_plan_expands_complete_cartesian_matrix():
    report = build_experiment_plan(sample_plan())
    assert report["status"] == "PASS"
    assert report["expected_runs"] == 8
    assert len(report["run_matrix"]) == 8
    assert report["coverage"]["cartesian_coverage_complete"] is True
    assert len(set(report["expected_run_ids"])) == 8
    assert report["jobs_launched"] is False
    assert report["execution_connected"] is False
    assert report["automatic_model_selection"] is False


def test_plan_is_deterministic_for_same_input():
    plan = sample_plan()
    first = build_experiment_plan(plan)
    second = build_experiment_plan(copy.deepcopy(plan))
    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["expected_run_ids"] == second["expected_run_ids"]
    assert first["run_matrix"] == second["run_matrix"]


def test_changed_seed_changes_plan_fingerprint_and_run_ids():
    original = build_experiment_plan(sample_plan())
    changed = sample_plan()
    changed["seeds"] = [11, 33]
    revised = build_experiment_plan(changed)
    assert original["plan_sha256"] != revised["plan_sha256"]
    assert set(original["expected_run_ids"]).isdisjoint(revised["expected_run_ids"])


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda p: p.update({"dataset_sha256": "bad"}), "dataset_sha256_invalid"),
        (lambda p: p.update({"seeds": [11, 11]}), "seeds_must_be_unique"),
        (lambda p: p.update({"folds": ["fold-1", "fold-1"]}), "folds_must_be_unique"),
        (
            lambda p: p.update({"evaluation_windows": [
                {"window_id": "bad", "start": "2026-02-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"}
            ]}),
            "evaluation_window_order_invalid:bad",
        ),
        (
            lambda p: p.update({"evaluation_windows": [
                {"window_id": "local", "start": "2026-01-01T00:00:00+05:30", "end": "2026-02-01T00:00:00+05:30"}
            ]}),
            "evaluation_window_must_be_utc:local",
        ),
    ],
)
def test_invalid_plan_fails_closed(mutation, error):
    plan = sample_plan()
    mutation(plan)
    with pytest.raises(ValueError, match=error.replace(":", r"\:")):
        build_experiment_plan(plan)


def test_run_limit_is_enforced():
    plan = sample_plan()
    plan["seeds"] = list(range(101))
    plan["folds"] = [f"fold-{index}" for index in range(51)]
    plan["evaluation_windows"] = [
        {"window_id": "only", "start": "2026-01-01T00:00:00Z", "end": "2026-02-01T00:00:00Z"}
    ]
    with pytest.raises(ValueError, match="planned_run_limit_exceeded"):
        build_experiment_plan(plan)

@pytest.mark.parametrize(
    "raw",
    [
        b'{"plan_id":"first","plan_id":"shadowed"}',
        b'{"metric":NaN}',
        b'{"metric":Infinity}',
        b'{"metric":-Infinity}',
    ],
)
def test_plan_cli_rejects_ambiguous_or_nonstandard_json(tmp_path, capsys, raw):
    from app.research_experiment_plan import main

    source = tmp_path / "input.json"
    source.write_bytes(raw)
    output = tmp_path / "plan.json"

    result = main(["--input", str(source), "--output", str(output)])

    assert result == 2
    assert not output.exists()
    assert '"status": "FAIL"' in capsys.readouterr().out


def test_plan_cli_requires_json_object_root(tmp_path, capsys):
    from app.research_experiment_plan import main

    source = tmp_path / "input.json"
    source.write_text("[1, 2, 3]", encoding="utf-8")
    output = tmp_path / "plan.json"

    result = main(["--input", str(source), "--output", str(output)])

    assert result == 2
    assert not output.exists()

