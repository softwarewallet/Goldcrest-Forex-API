# Goldcrest Forex API

Standalone AI market-analysis API for GoldcrestForex. Prediction and analysis only: no broker credentials and no order-placement endpoints.

## Stack
- Python + FastAPI + Pydantic
- llama.cpp OpenAI-compatible API, default `http://192.168.29.80:8080`
- SQLite through SQLAlchemy
- pytest

GoldcrestForex remains responsible for execution, sizing, and all risk gates. GoldcrestAI is developed and validated independently before any integration is authorized.

## Development setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Replace API_KEY with a newly generated secret before using the API.
python -m pytest -q
uvicorn app.main:app --host 127.0.0.1 --port 8090
```

The API bind address and port are set by the Uvicorn command, not by the `GOLDCREST_BIND_HOST` / `GOLDCREST_PORT` settings in `.env`. For trusted-LAN access, run Uvicorn with `--host 0.0.0.0 --port 8090` and configure the host firewall to permit port 8090 only from the required private subnet. Never expose this API or llama.cpp directly to the public Internet. The production systemd service should use explicit, reviewed bind and port arguments.

## Routes
- `GET /api/v1/health` — basic database and model-listing status
- `GET /api/v1/readiness/inference` — protected live completion probe; performs actual inference
- `GET /api/v1/models` — protected model listing
- `POST /api/v1/predictions` — protected prediction analysis
- `GET /api/v1/predictions/{prediction_id}` — protected prediction lookup
- `POST /api/v1/outcomes` — protected outcome recording
- `GET /api/v1/performance` — protected counts; no directional performance claims yet
- `GET /api/v1/audit` — protected audit history

Protected routes use `X-API-Key`. Set `LLAMA_MODEL_ID` to the exact ID returned by llama.cpp `/v1/models`, or leave it empty to auto-select only when exactly one model is served. Inference currently uses llama.cpp's `/v1/completions` endpoint and parses `choices[0].text`.

## Live inference smoke test
1. Confirm the llama.cpp server is reachable at `LLAMA_BASE_URL` and `GET /v1/models` returns the expected model ID.
2. Start the API with the same `.env` configuration.
3. Set `API_KEY` to a fresh random secret (at least 32 characters); never commit or share it.
4. Call the protected inference readiness endpoint from the API host:
   ```bash
   curl -i http://127.0.0.1:8090/api/v1/readiness/inference \
     -H "X-API-Key: $API_KEY"
   ```
5. Expect HTTP 200 with `{"status":"ok","model_id":"...","completion_valid":true}`. A 503 indicates inference or output validation failed; inspect the API logs without exposing secrets or market payloads.

## Prediction status
The initial prediction service is scaffolding, not a validated trading strategy. It deliberately returns `NO_TRADE` even when the LLM suggests BUY or SELL, until a quantitative rule is validated. Model self-reported confidence is not used as a calibrated probability. No broker credentials are accepted or stored.

See `docs/phase-1-audit.md` for current audit findings and remaining work.


## Resource-aware Ubuntu runtime

For constrained laptops, run the lightweight environment preflight before starting research:

```bash
python -m app.runtime_diagnostics
python -m app.runtime_diagnostics --json
```

XGBoost and LightGBM use a conservative RAM-aware CPU thread limit. Override it for
controlled benchmarks with `GOLDCREST_ML_THREADS=1` (or another positive integer).
See `docs/resource-aware-runtime.md` for limits, diagnostics, and safeguards.
