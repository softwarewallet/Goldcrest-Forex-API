"""Read-only cTrader Open API historical OHLC ingestion for offline research.

Credentials are read only from environment variables. This module only sends application
auth, account auth, symbol-list and trendbar-history requests; it has no order API imports.
The default endpoint is DEMO. All downloaded data is marked non-actionable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from dotenv import dotenv_values

from app.ctrader_history import normalize_ctrader_trendbars

_ALLOWED_PERIODS = {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_ENV_NAMES = (
    "CTRADER_CLIENT_ID",
    "CTRADER_CLIENT_SECRET",
    "CTRADER_ACCESS_TOKEN",
    "CTRADER_ACCOUNT_ID",
)


@dataclass(frozen=True)
class CTraderResearchConfig:
    client_id: str
    client_secret: str
    access_token: str
    account_id: int
    host: str = "demo"


def load_config(environ: dict[str, str] | None = None) -> CTraderResearchConfig:
    """Read credentials from the environment or repository-root .env, without printing or persisting them.

    Explicit mappings are used unchanged for deterministic tests and callers. In normal CLI
    use, non-empty variables already provided by the process take precedence over .env values.
    """
    if environ is None:
        dotenv_path = _PROJECT_ROOT / ".env"
        file_values = dotenv_values(dotenv_path=dotenv_path, encoding="utf-8")
        # Keep the documented precedence rule: non-empty process values win, but an
        # exported empty variable must not shadow a valid value in the repository .env.
        # Copy only cTrader settings into the process environment; never print them.
        for name in (*_ENV_NAMES, "CTRADER_HOST"):
            process_value = os.environ.get(name, "")
            file_value = file_values.get(name)
            if not process_value.strip() and isinstance(file_value, str) and file_value.strip():
                os.environ[name] = file_value
        source = os.environ
    else:
        source = environ
    missing = [name for name in _ENV_NAMES if not source.get(name, "").strip()]
    if missing:
        raise ValueError("Missing required environment variables: " + ", ".join(missing))
    try:
        account_id = int(source["CTRADER_ACCOUNT_ID"])
    except (TypeError, ValueError) as exc:
        raise ValueError("CTRADER_ACCOUNT_ID must be an integer") from exc
    if account_id <= 0:
        raise ValueError("CTRADER_ACCOUNT_ID must be positive")
    host = source.get("CTRADER_HOST", "demo").strip().lower()
    if host != "demo":
        raise ValueError("Research history ingestion is restricted to DEMO; set CTRADER_HOST=demo")
    return CTraderResearchConfig(
        client_id=source["CTRADER_CLIENT_ID"].strip(),
        client_secret=source["CTRADER_CLIENT_SECRET"].strip(),
        access_token=source["CTRADER_ACCESS_TOKEN"].strip(),
        account_id=account_id,
        host=host,
    )


def validate_request(symbol: str, timeframe: str, lookback_days: int) -> tuple[str, str, int]:
    normalized_symbol = symbol.strip().upper()
    normalized_period = timeframe.strip().upper()
    if not normalized_symbol or not re.fullmatch(r"[A-Z0-9._/-]{3,32}", normalized_symbol):
        raise ValueError("symbol must be a valid non-empty instrument name")
    if normalized_period not in _ALLOWED_PERIODS:
        raise ValueError(f"timeframe must be one of {sorted(_ALLOWED_PERIODS)}")
    if isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or not 1 <= lookback_days <= 3650:
        raise ValueError("lookback_days must be an integer from 1 to 3650")
    return normalized_symbol, normalized_period, lookback_days


def _safe_failure(failure: Any) -> str:
    """Return a short diagnostic without dumping request objects or credential-bearing state."""
    value = getattr(failure, "value", failure)
    return type(value).__name__


def download_ctrader_history(
    *,
    symbol: str,
    timeframe: str,
    lookback_days: int,
    output_csv: str | Path,
    config: CTraderResearchConfig | None = None,
    client_factory: Callable[..., Any] | None = None,
    reactor_instance: Any | None = None,
    sdk: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch historical trendbars with cTrader OpenApiPy and persist a research snapshot.

    Dependency injection is provided for tests. A real network connection is created only
    when this function is explicitly called.
    """
    instrument, period, days = validate_request(symbol, timeframe, lookback_days)
    selected_config = config or load_config()
    if selected_config.host != "demo":
        raise ValueError("only DEMO cTrader history ingestion is permitted")
    if sdk is None:
        from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAApplicationAuthReq,
            ProtoOAAccountAuthReq,
            ProtoOASymbolsListReq,
            ProtoOAGetTrendbarsReq,
        )
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbarPeriod
        from twisted.internet import reactor as twisted_reactor
        sdk = {
            "Client": Client,
            "Protobuf": Protobuf,
            "TcpProtocol": TcpProtocol,
            "EndPoints": EndPoints,
            "ProtoOAApplicationAuthReq": ProtoOAApplicationAuthReq,
            "ProtoOAAccountAuthReq": ProtoOAAccountAuthReq,
            "ProtoOASymbolsListReq": ProtoOASymbolsListReq,
            "ProtoOAGetTrendbarsReq": ProtoOAGetTrendbarsReq,
            "ProtoOATrendbarPeriod": ProtoOATrendbarPeriod,
            "reactor": twisted_reactor,
        }

    loop = reactor_instance or sdk["reactor"]
    factory = client_factory or sdk["Client"]
    endpoint = sdk["EndPoints"].PROTOBUF_DEMO_HOST
    client = factory(endpoint, sdk["EndPoints"].PROTOBUF_PORT, sdk["TcpProtocol"])
    result: dict[str, Any] = {}
    failure_state: dict[str, str] = {}

    def stop_loop() -> None:
        if getattr(loop, "running", False):
            loop.stop()

    def fail(failure: Any) -> Any:
        failure_state["error_type"] = _safe_failure(failure)
        stop_loop()
        return failure

    def get_bars(symbol_id: int) -> None:
        request = sdk["ProtoOAGetTrendbarsReq"]()
        request.ctidTraderAccountId = selected_config.account_id
        request.symbolId = int(symbol_id)
        request.period = getattr(sdk["ProtoOATrendbarPeriod"], period)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        request.fromTimestamp = int(start_time.timestamp() * 1000)
        request.toTimestamp = int(end_time.timestamp() * 1000)
        deferred = client.send(request)
        deferred.addCallbacks(on_bars, fail)

    def on_bars(message: Any) -> Any:
        try:
            response = sdk["Protobuf"].extract(message)
            trendbars = list(response.trendbar)
            if not trendbars:
                raise ValueError("cTrader returned no trendbars for this symbol/timeframe/window")
            snapshot = normalize_ctrader_trendbars(
                trendbars,
                symbol=instrument,
                timeframe=period,
                output_csv=output_csv,
            )
            retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            snapshot["retrieved_at_utc"] = retrieved_at
            snapshot["host"] = "demo"
            snapshot["request_lookback_days"] = days
            manifest_path = Path(output_csv).with_suffix(Path(output_csv).suffix + ".manifest.json")
            manifest_path.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            result.update(snapshot)
        except Exception as exc:
            failure_state["error_type"] = type(exc).__name__
        finally:
            stop_loop()
        return message

    def on_symbols(message: Any) -> Any:
        try:
            response = sdk["Protobuf"].extract(message)
            matches = [
                item for item in response.symbol
                if str(getattr(item, "symbolName", "")).strip().upper() == instrument
            ]
            if not matches:
                raise ValueError("requested symbol was not found in the account symbol list")
            if len(matches) > 1:
                raise ValueError("requested symbol is ambiguous in the account symbol list")
            get_bars(int(matches[0].symbolId))
        except Exception as exc:
            failure_state["error_type"] = type(exc).__name__
            stop_loop()
        return message

    def on_account_auth(message: Any) -> Any:
        request = sdk["ProtoOASymbolsListReq"]()
        request.ctidTraderAccountId = selected_config.account_id
        request.includeArchivedSymbols = False
        client.send(request).addCallbacks(on_symbols, fail)
        return message

    def on_application_auth(message: Any) -> Any:
        request = sdk["ProtoOAAccountAuthReq"]()
        request.ctidTraderAccountId = selected_config.account_id
        request.accessToken = selected_config.access_token
        client.send(request).addCallbacks(on_account_auth, fail)
        return message

    def connected(connected_client: Any) -> None:
        request = sdk["ProtoOAApplicationAuthReq"]()
        request.clientId = selected_config.client_id
        request.clientSecret = selected_config.client_secret
        connected_client.send(request).addCallbacks(on_application_auth, fail)

    def disconnected(_client: Any, reason: Any) -> None:
        if not result and not failure_state:
            failure_state["error_type"] = "DisconnectedBeforeHistoryResponse"
        stop_loop()

    def on_message(_client: Any, message: Any) -> None:
        # Deferred callbacks own expected responses. Ignore unsolicited messages and heartbeats.
        return None

    client.setConnectedCallback(connected)
    client.setDisconnectedCallback(disconnected)
    client.setMessageReceivedCallback(on_message)
    client.startService()
    loop.run()

    if failure_state:
        raise RuntimeError(
            "cTrader research history download failed (diagnostic type: "
            + failure_state["error_type"] + ")"
        )
    if not result:
        raise RuntimeError("cTrader research history download ended without a result")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download cTrader DEMO historical OHLC for offline research.")
    parser.add_argument("--symbol", required=True, help="Exact cTrader account symbol name, e.g. EURUSD")
    parser.add_argument("--timeframe", required=True, choices=tuple(sorted(_ALLOWED_PERIODS)))
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = download_ctrader_history(
            symbol=args.symbol,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
            output_csv=args.output_csv,
        )
    except Exception as exc:
        print(f"cTrader research ingestion failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(json.dumps({
        "scope": result["scope"],
        "symbol": result["symbol"],
        "timeframe": result["timeframe"],
        "rows": result["rows"],
        "snapshot_sha256": result["snapshot_sha256"],
        "host": result["host"],
        "actionable": False,
        "output_csv": args.output_csv,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
