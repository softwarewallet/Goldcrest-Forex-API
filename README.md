# Goldcrest Forex API

Standalone AI market-analysis API for GoldcrestForex. Prediction and analysis only: no broker credentials and no order-placement endpoints.

## Initial stack
- Python + FastAPI + Pydantic
- llama.cpp OpenAI-compatible API, default `http://192.168.29.80:8080`
- SQLite through SQLAlchemy
- pytest

GoldcrestForex remains responsible for execution, sizing, and all risk gates.

## Development setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Replace API_KEY with a newly generated secret before using the API.
uvicorn app.main:app --host 127.0.0.1 --port 8090
```

Default bind is loopback. For LAN access, configure `GOLDCREST_BIND_HOST=0.0.0.0` and firewall rules so only trusted clients can reach port 8090. Never expose this API or llama.cpp directly to the public Internet.

## Routes
- `GET /api/v1/health`
- `GET /api/v1/models`
- `POST /api/v1/predictions`
- `GET /api/v1/predictions/{prediction_id}`
- `POST /api/v1/outcomes`
- `GET /api/v1/performance`
- `GET /api/v1/audit`

Protected routes use `X-API-Key`. Set `LLAMA_MODEL_ID` to the exact ID returned by llama.cpp `/v1/models`, or leave it empty for automatic selection when exactly one model is served.

The initial prediction service is a scaffolding baseline, not a validated trading strategy. It is fail-safe and deliberately returns `NO_TRADE` until a quantitative rule is validated. It must not invent calibrated probabilities. No broker credentials are accepted or stored.
