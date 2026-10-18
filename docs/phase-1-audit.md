# Phase 1 Code Review and Test Audit

Date: 2026-10-18
Scope: standalone GoldcrestAI API foundation only. No GoldcrestForex integration or broker execution changes.

## Verified from repository and CI

- FastAPI routes are declared for health, model listing, predictions, outcomes, performance, and audit records.
- Protected routes use an API key checked with constant-time comparison.
- Market request schemas reject extra fields, invalid OHLC values, unordered/duplicate candle timestamps, and timezone-naive timestamps.
- Prediction service rejects stale/future snapshots and returns `NO_TRADE` for every current prediction, including when the LLM suggests BUY/SELL.
- Broker credentials and order-placement routes are not part of this repository.
- The original GitHub Actions run failed during test collection with `ModuleNotFoundError: No module named 'app'`; the final test configuration now sets the pytest project import path and uses `python -m pytest`.
- The CI test job uses a dedicated test-only API key; it must never be used for deployment.

## Fixes in this audit PR

1. Add pytest configuration so `app` imports consistently in CI and local runs.
2. Make CI authentication settings deterministic.
3. Remove the duplicate `httpx` dependency.
4. Stub model-list access in the health test so the test suite does not require a live llama.cpp server.
5. Add tests that assert LLM BUY suggestions remain `NO_TRADE` and inference failures fail safe.
6. Quote `APP_NAME` in `.env.example`, avoiding the Bash `source .env` issue where `Forex` was treated as a command.
7. Correct README instructions: Uvicorn's explicit `--host` and `--port` arguments control binding; the current settings fields alone do not change Uvicorn's listener.

## Findings that remain open

### High — End-to-end inference is not certified
The health endpoint verifies the model-list endpoint, not a real chat-completions generation. A healthy status does not prove `/v1/chat/completions` works with the loaded model/template. Add a separately controlled inference readiness check and an integration test against the actual llama.cpp server before claiming inference is certified.

### High — Predictions are not actionable by design
The current prediction service always returns `NO_TRADE`. This is correct for the current safe baseline. A quantitative decision rule, calibrated probability model, historical testing, and walk-forward/out-of-sample validation have not been completed.

### High — Request idempotency needs payload binding and concurrency handling
The prediction endpoint returns an existing result solely by `request_id`. Reusing an ID with different market data can silently return a prediction for the earlier request. Concurrent requests with the same new ID may race on the unique constraint. Bind IDs to a canonical payload hash and handle unique-constraint conflicts explicitly.

### Medium — SQLite timestamps may lose timezone metadata
SQLite can return values stored in `DateTime(timezone=True)` columns without timezone information. Normalize persisted timestamps to UTC on response and add a persistence round-trip test.

### Medium — Database schema evolution is not managed
`Base.metadata.create_all` creates missing tables but does not migrate existing tables when columns or constraints change. Introduce explicit migrations before schema evolution.

### Medium — Outcome scoring rules are not yet sufficient for performance claims
Outcomes can currently be recorded using a label and timestamp, but the API does not establish a complete, reproducible price-based scoring protocol. Do not publish accuracy or profitability metrics until definitions, horizon eligibility, and costs are specified.

### Medium — Health route is a dependency check, not an inference test
It lists models and marks inference as available if at least one model is listed. It does not perform generation and should not be interpreted as proof of end-to-end prediction readiness.

### Low — Performance endpoint loads all rows into memory
It currently loads all prediction and outcome records to count them. Replace this with database-side aggregate counts before the history grows.

### Low — Request and stored-output limits need review
Review maximum request body size, outcome details size, raw model-output retention, audit retention, and rate limiting before any broader deployment. Keep the API restricted to trusted LAN clients.

## Exit criteria for Phase 1

- CI green on the final PR head.
- Local test suite passes.
- The actual llama.cpp chat-completions endpoint is verified.
- Prediction and persistence edge cases above are either fixed or covered by explicit tests.
- Remaining limitations are documented; no claim of production-ready signals is made.

GoldcrestAI remains disconnected from GoldcrestForex until the user explicitly authorizes integration.
