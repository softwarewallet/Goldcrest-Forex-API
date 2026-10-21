# Resource-aware research runtime

The local Ubuntu laptop is a constrained development target, not a benchmark host.
This profile keeps XGBoost and LightGBM training CPU-thread usage conservative based on
detected system RAM, with an explicit override for controlled experiments.

## Inspect the environment

From the repository root, with the project environment activated:

```bash
python -m app.runtime_diagnostics
python -m app.runtime_diagnostics --json > reports/runtime-diagnostics.json
```

The command uses Python's standard library for package discovery, reads Linux memory
information, and queries `nvidia-smi` if available. It does not import large frameworks,
contact the LLM gateway, check model-serving health, collect secrets, or claim that GPU
acceleration is available merely because a GPU is installed.

## Thread policy

- Up to 8 GiB RAM: at most 2 ML CPU threads by default; run one model-training process
  at a time and avoid loading multiple large models simultaneously.
- More than 8 GiB and up to 16 GiB: at most 4 threads by default.
- Above 16 GiB: at most 8 threads by default.
- Defaults are always capped at the detected logical CPU count.
- Set `GOLDCREST_ML_THREADS=1` or another positive integer to request a different
  thread count; requests above the CPU count are clamped. Invalid values fail clearly.

The XGBoost and LightGBM wrappers use this setting unless a model fit call explicitly
overrides `n_jobs` through its existing parameter argument. These are conservative
starting values, not a promise of best wall-clock performance. Benchmark on genuine
datasets and measure peak memory before increasing concurrency.

## Existing installed tools

The preflight checks Python module discovery for NumPy, Pandas, PyTorch, TA-Lib,
scikit-learn, XGBoost, LightGBM, and Optuna. A module being found does not prove that
its binary dependencies or drivers work. PyTorch, TA-Lib, and Optuna are not automatically
introduced into model selection by this change; add them to a research experiment only
with a defined feature/model hypothesis and tests.

The names Vega, Nova, and Titan are listed as unverified because their exact distributions,
model IDs, or runtimes are not established by those names alone. Do not silently assume
they are Python packages or model endpoints. Record their exact package/model IDs before
adding integrations.

## Research safeguards

This utility does not change model labels, features, walk-forward splits, provenance,
calibration, model selection, prediction API behavior, or the fail-closed
`actionable=false` contract. It does not claim predictive accuracy or profitability.
