"""Resource-aware defaults for offline research on constrained machines.

This module uses only the Python standard library so it can be imported before
heavy ML libraries. It never enables GPU execution or changes research semantics.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


_GIB = 1024 ** 3


def _read_memory_bytes() -> tuple[int | None, int | None]:
    """Return (total, available) bytes using Linux procfs, then sysconf fallback."""
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                token = rest.strip().split()[0]
                values[key] = int(token) * 1024
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if total:
            return total, available
    except (OSError, ValueError, IndexError):
        pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages), None
    except (AttributeError, OSError, ValueError):
        return None, None


@dataclass(frozen=True)
class ResourceProfile:
    cpu_count: int
    total_memory_bytes: int | None
    available_memory_bytes: int | None
    ml_threads: int
    profile_name: str
    recommendations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["total_memory_gib"] = (
            round(self.total_memory_bytes / _GIB, 2)
            if self.total_memory_bytes is not None else None
        )
        result["available_memory_gib"] = (
            round(self.available_memory_bytes / _GIB, 2)
            if self.available_memory_bytes is not None else None
        )
        result["recommendations"] = list(self.recommendations)
        return result


def build_resource_profile(
    *,
    cpu_count: int | None = None,
    total_memory_bytes: int | None = None,
    available_memory_bytes: int | None = None,
    thread_override: str | int | None = None,
) -> ResourceProfile:
    """Build a conservative CPU-thread profile; explicit overrides are CPU-clamped."""
    detected_cpus = os.cpu_count() or 1
    cpus = detected_cpus if cpu_count is None else cpu_count
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
        raise ValueError("cpu_count must be a positive integer")

    if total_memory_bytes is None:
        detected_total, detected_available = _read_memory_bytes()
        total = detected_total
        available = available_memory_bytes if available_memory_bytes is not None else detected_available
    else:
        total = total_memory_bytes
        available = available_memory_bytes
    if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total <= 0):
        raise ValueError("total_memory_bytes must be a positive integer or None")
    if available is not None and (
        isinstance(available, bool) or not isinstance(available, int) or available < 0
    ):
        raise ValueError("available_memory_bytes must be a non-negative integer or None")

    if thread_override is None:
        override_text = os.getenv("GOLDCREST_ML_THREADS", "").strip()
        override: int | None = int(override_text) if override_text else None
    else:
        if isinstance(thread_override, bool):
            raise ValueError("thread_override must be a positive integer")
        try:
            override = int(thread_override)
        except (TypeError, ValueError) as exc:
            raise ValueError("thread_override must be a positive integer") from exc
    if override is not None and override < 1:
        raise ValueError("GOLDCREST_ML_THREADS must be a positive integer")

    if total is None:
        profile_name = "conservative_unknown_memory"
        default_threads = min(2, cpus)
        notes = (
            "Memory capacity could not be detected; keep experiments small.",
            "Use one experiment at a time and monitor system memory.",
        )
    elif total <= 8 * _GIB:
        profile_name = "constrained_8gib"
        default_threads = min(2, cpus)
        notes = (
            "Use small chronological folds and one training process at a time.",
            "Prefer CPU inference with a small quantized LLM; avoid loading multiple models together.",
            "Close memory-heavy applications before full historical research runs.",
        )
    elif total <= 16 * _GIB:
        profile_name = "balanced_16gib"
        default_threads = min(4, cpus)
        notes = (
            "Run one model-training job at a time unless memory monitoring shows headroom.",
            "Keep the LLM context length modest when running it beside tabular research.",
        )
    else:
        profile_name = "throughput"
        default_threads = min(8, cpus)
        notes = (
            "Increase concurrency only after measuring peak memory and runtime.",
            "Keep walk-forward and dataset provenance controls unchanged.",
        )

    threads = min(override, cpus) if override is not None else default_threads
    if override is not None and override > cpus:
        notes = notes + (f"Requested thread count was clamped to detected CPU count ({cpus}).",)
    return ResourceProfile(
        cpu_count=cpus,
        total_memory_bytes=total,
        available_memory_bytes=available,
        ml_threads=threads,
        profile_name=profile_name,
        recommendations=notes,
    )


def get_ml_thread_count() -> int:
    """Return the current machine's safe default, overridable by GOLDCREST_ML_THREADS."""
    return build_resource_profile().ml_threads
