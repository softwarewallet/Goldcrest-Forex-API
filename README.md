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
- `GET /api/v1/health`
- `GET /api/v1/models`
- `POST /api/v1/predictions`
- `GET /api/v1/predictions/{prediction_id}`
- `POST /api/v1/outcomes`
- `GET /api/v1/performance`
- `GET /api/v1/audit`

Protected routes use `X-API-Key`. Set `LLAMA_MODEL_ID` to the exact ID returned by llama.cpp `/v1/models`, or leave it empty to auto-select only when exactly one model is served.

## Prediction status
The initial prediction service is scaffolding, not a validated trading strategy. It deliberately returns `NO_TRADE` even when the LLM suggests BUY or SELL, until a quantitative rule is validated. Model self-reported confidence is not used as a calibrated probability. No broker credentials are accepted or stored.

See `docs/phase-1-audit.md` for current audit findings and remaining work.
