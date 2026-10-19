# GoldcrestAI Neural Signal Console

A standalone React + TypeScript dashboard for the independent GoldcrestAI research API. It is intentionally isolated from GoldcrestForex and contains no broker or order-placement client.

## Run locally

```bash
cd dashboard
npm install
npm run dev
```

The Vite development server listens on all interfaces. Restrict access with your host firewall; do not expose the API or model server directly to the public Internet.

## Connect

Use the settings control to enter the API base URL and `X-API-Key`. The URL is saved in local storage; the API key is held in session storage only and is not sent to any third party by this app. The dashboard polls health, performance, prediction history, and audit endpoints every 15 seconds. The **Test Inference** control separately calls the protected readiness endpoint to run an actual model completion and validate its JSON output; it is not called on every poll.

## Signal-flow integrity

The flow animation is active only while a dashboard API poll is in flight. Source/model/output labels report observed API state, not a simulated continuous market feed. The UI never injects demo predictions. Empty results and connection failures are explicitly shown.

All displayed predictions are research-only. The API intentionally gates decisions to `NO_TRADE` until a quantitative strategy is validated. This dashboard has no execution controls, broker credentials, or order routes.

## API contract

The dashboard uses:
- `GET /api/v1/health` (unprotected health status)
- `GET /api/v1/performance`
- `GET /api/v1/predictions?limit=30`
- `GET /api/v1/audit?limit=12`

The last three require `X-API-Key`.
