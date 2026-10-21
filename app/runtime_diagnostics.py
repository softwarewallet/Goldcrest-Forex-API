"""Lightweight environment inventory for Goldcrest's offline research runtime.

Run with: python -m app.runtime_diagnostics [--json]
The command does not import ML frameworks, contact network services, or read secrets.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from typing import Any

from app.resource_profile import build_resource_profile


_PYTHON_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "PyTorch": "torch",
    "TA-Lib": "talib",
    "scikit-learn": "sklearn",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Optuna": "optuna",
}


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _nvidia_gpus() -> list[dict[str, str]] | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if parts and parts[0]:
            gpus.append({
                "name": parts[0],
                "memory": parts[1] if len(parts) > 1 else "unknown",
            })
    return gpus


def collect_runtime_diagnostics() -> dict[str, Any]:
    profile = build_resource_profile()
    base_url = os.getenv("LLAMA_BASE_URL", "").strip()
    model_id = os.getenv("LLAMA_MODEL_ID", "").strip()
    return {
        "scope": "OFFLINE_RESEARCH_ONLY",
        "actionable": False,
        "python_packages": {
            label: {"module": module, "available": _module_available(module)}
            for label, module in _PYTHON_MODULES.items()
        },
        "hardware": {
            "resource_profile": profile.to_dict(),
            "nvidia_gpus": _nvidia_gpus(),
        },
        "llm_gateway": {
            "base_url_configured": bool(base_url),
            "model_id_configured": bool(model_id),
            "network_checked": False,
            "note": "This inventory does not connect to the gateway or verify the served model.",
        },
        "unverified_components": {
            "names": ["Vega", "Nova", "Titan"],
            "note": "Names are recorded as user-reported; identify their exact package/model IDs before integration.",
        },
        "notes": [
            "Package discovery does not prove imports, drivers, or GPU acceleration work.",
            "A missing nvidia-smi result is not proof that no GPU exists.",
            "No secrets, credentials, broker settings, or market data are collected.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the local research runtime without starting services.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)
    report = collect_runtime_diagnostics()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        profile = report["hardware"]["resource_profile"]
        print("Goldcrest-Forex-API runtime preflight (research-only)")
        print(f"Profile: {profile['profile_name']}")
        print(f"CPU threads for ML: {profile['ml_threads']} / {profile['cpu_count']}")
        print(f"RAM: {profile['total_memory_gib']} GiB total; {profile['available_memory_gib']} GiB available")
        for label, item in report["python_packages"].items():
            print(f"{label}: {'available' if item['available'] else 'not detected'}")
        gpus = report["hardware"]["nvidia_gpus"]
        print(f"NVIDIA GPU query: {gpus if gpus is not None else 'unavailable (not verified)'}")
        for note in profile["recommendations"]:
            print(f"- {note}")
        print("Actionable: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
