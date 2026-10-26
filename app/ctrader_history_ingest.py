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
class CTraderApplicationConfig:
    client_id: str
    client_secret: str
    access_token: str
    host: str = "demo"


@dataclass(frozen=True)
class CTraderResearchConfig:
    client_id: str
    client_secret: str
    access_token: str
    account_id: int
    host: str = "demo"


def _configuration_source(environ: dict[str, str] | None) -> Any:
    """Return explicit values or process environment populated from non-empty .env settings."""
    if environ is not None:
        return environ
    dotenv_path = _PROJECT_ROOT / ".env"
    file_values = dotenv_values(dotenv_path=dotenv_path, encoding="utf-8")
    # Non-empty process values win; empty exported values fall back to the repository .env.
    for name in (*_ENV_NAMES, "CTRADER_HOST"):
        process_value = os.environ.get(name, "")
        file_value = file_values.get(name)
        if not process_value.strip() and isinstance(file_value, str) and file_value.strip():
            os.environ[name] = file_value
    return os.environ


def load_application_config(environ: dict[str, str] | None = None) -> CTraderApplicationConfig:
    """Load only application credentials and DEMO host; account ID is deliberately not required."""
    source = _configuration_source(environ)
    required = ("CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET", "CTRADER_ACCESS_TOKEN")
    missing = [name for name in required if not source.get(name, "").strip()]
    if missing:
        raise ValueError("Missing required environment variables: " + ", ".join(missing))
    host = source.get("CTRADER_HOST", "demo").strip().lower()
    if host != "demo":
        raise ValueError("Research account discovery is restricted to DEMO; set CTRADER_HOST=demo")
    return CTraderApplicationConfig(
        client_id=source["CTRADER_CLIENT_ID"].strip(),
        client_secret=source["CTRADER_CLIENT_SECRET"].strip(),
        access_token=source["CTRADER_ACCESS_TOKEN"].strip(),
        host=host,
    )


def load_config(environ: dict[str, str] | None = None) -> CTraderResearchConfig:
    """Read credentials from the environment or repository-root .env, without printing or persisting them.

    Explicit mappings are used unchanged for deterministic tests and callers. In normal CLI
    use, non-empty variables already provided by the process take precedence over .env values.
    """
    source = _configuration_source(environ)
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


def _safe_api_error_diagnostic(response: Any) -> str | None:
    """Expose only a recognized cTrader error type and bounded error code, never descriptions."""
    response_type = type(response).__name__
    if response_type not in {"ProtoOAErrorRes", "ProtoErrorRes"}:
        return None
    raw_code = getattr(response, "errorCode", None) or getattr(response, "error_code", None) or "UNKNOWN"
    code = str(raw_code)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", code):
        code = "UNRECOGNIZED"
    return f"{response_type} code={code}"


# Keep each request inside conservative cTrader period-specific timestamp limits.
# The bounds are intentionally below the documented maxima to tolerate rounding differences.
_MINUTE_MS = 60_000
_TREND_BAR_PAGE_SIZE = 1000
_TREND_BAR_MAX_PAGES = 10_000
_SEND_MAX_ATTEMPTS = 3
_SEND_RETRY_BASE_DELAY_SECONDS = 2
_SEND_RESPONSE_TIMEOUT_SECONDS = 20
_TRANSIENT_SEND_ERRORS = {
    "TimeoutError",
    "TCPTimedOutError",
    "ConnectionLost",
    "ConnectionDone",
    "ConnectError",
    "ConnectionRefusedError",
}
_MAX_WINDOW_DAYS = {
    "M1": 34,
    "M5": 34,
    "M15": 240,
    "M30": 240,
    "H1": 240,
    "H4": 360,
    "D1": 360,
}


def _trendbar_timestamp_ms(bar: Any) -> int:
    """Return a cTrader trendbar's UTC opening timestamp in milliseconds."""
    value = bar.get("utcTimestampInMinutes") if isinstance(bar, dict) else getattr(bar, "utcTimestampInMinutes", None)
    if value is not None:
        return int(value) * _MINUTE_MS
    value = bar.get("timestamp_utc", bar.get("timestamp")) if isinstance(bar, dict) else getattr(
        bar, "timestamp_utc", getattr(bar, "timestamp", None)
    )
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trendbar timestamp must include timezone information")
        return int(value.timestamp() * 1000)
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("trendbar timestamp must include timezone information")
        return int(parsed.timestamp() * 1000)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Generic adapters may provide Unix epoch seconds.
        return int(float(value) * 1000)
    raise ValueError("trendbar does not expose a supported timestamp")


def _read_wire_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(payload) and shift < 70:
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("invalid protobuf varint")


def _extract_optional_bool_wire_field(payload: bytes, field_number: int) -> bool | None:
    """Read an optional bool from protobuf wire bytes when an older SDK schema omits it."""
    offset = 0
    try:
        while offset < len(payload):
            tag, offset = _read_wire_varint(payload, offset)
            number, wire_type = tag >> 3, tag & 0x07
            if number == 0:
                return None
            if number == field_number:
                if wire_type != 0:
                    return None
                value, _offset = _read_wire_varint(payload, offset)
                return bool(value)
            if wire_type == 0:
                _value, offset = _read_wire_varint(payload, offset)
            elif wire_type == 1:
                offset += 8
            elif wire_type == 2:
                length, offset = _read_wire_varint(payload, offset)
                offset += length
            elif wire_type == 5:
                offset += 4
            else:
                return None
            if offset > len(payload):
                return None
    except (ValueError, IndexError):
        return None
    return None


def _response_has_more_trendbars(response: Any, message: Any | None = None) -> bool | None:
    """Read hasMore from the response or raw protobuf payload, even with an older SDK schema."""
    has_field = getattr(response, "HasField", None)
    if callable(has_field):
        try:
            if has_field("hasMore"):
                value = getattr(response, "hasMore")
                return bool(value) if isinstance(value, bool) else None
        except (ValueError, TypeError):
            pass
    value = getattr(response, "hasMore", None)
    if isinstance(value, bool):
        return value

    # ctrader-open-api==0.9.2's generated response class predates hasMore field 7.
    # The SDK still passes through the original ProtoMessage payload, so inspect only that
    # top-level optional bool rather than upgrading/vendor-modifying the generated schema.
    payload = getattr(message, "payload", None)
    if isinstance(payload, (bytes, bytearray)):
        return _extract_optional_bool_wire_field(bytes(payload), 7)
    return None


def _build_trendbar_windows(
    start_timestamp_ms: int,
    end_timestamp_ms: int,
    timeframe: str,
) -> list[tuple[int, int]]:
    """Partition a history interval into overlapping minute-aligned windows within cTrader limits."""
    if timeframe not in _MAX_WINDOW_DAYS:
        raise ValueError("unsupported timeframe for cTrader history windows")
    start = (int(start_timestamp_ms) // _MINUTE_MS) * _MINUTE_MS
    end = (int(end_timestamp_ms) // _MINUTE_MS) * _MINUTE_MS
    if start < 0 or end < start:
        raise ValueError("invalid historical time range")
    max_span = _MAX_WINDOW_DAYS[timeframe] * 24 * 60 * _MINUTE_MS
    windows: list[tuple[int, int]] = []
    window_start = start
    while window_start <= end:
        window_end = min(end, window_start + max_span)
        window_end = (window_end // _MINUTE_MS) * _MINUTE_MS
        windows.append((window_start, window_end))
        if window_end >= end:
            break
        # Reuse the boundary minute in the next window; final deduplication avoids gaps
        # from candle timestamps that sit exactly on a window boundary.
        window_start = window_end
    return windows


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

    def extract_response(message: Any) -> Any | None:
        response = sdk["Protobuf"].extract(message)
        diagnostic = _safe_api_error_diagnostic(response)
        if diagnostic:
            failure_state["error_type"] = diagnostic
            stop_loop()
            return None
        return response

    def fail(failure: Any, stage: str) -> None:
        # Consume the Deferred failure after preserving a sanitized stage/type diagnostic.
        # Returning the original Failure here leaves Twisted to log it as unhandled.
        failure_state["error_type"] = f"{stage}:{_safe_failure(failure)}"
        stop_loop()

    def send_with_retry(request: Any, callback: Callable[[Any], Any]) -> None:
        """Retry transient request timeouts with a bounded response timeout and backoff."""
        state = {"attempts": 0}
        stage = type(request).__name__

        def on_send_failure(failure: Any) -> None:
            value = getattr(failure, "value", failure)
            error_type = type(value).__name__
            if (
                error_type in _TRANSIENT_SEND_ERRORS
                and state["attempts"] < _SEND_MAX_ATTEMPTS
                and not failure_state
            ):
                delay = _SEND_RETRY_BASE_DELAY_SECONDS * state["attempts"]
                loop.callLater(delay, attempt)
                return None
            fail(failure, stage)
            return None

        def attempt() -> None:
            if failure_state:
                return
            state["attempts"] += 1
            try:
                deferred = client.send(
                    request,
                    responseTimeoutInSeconds=_SEND_RESPONSE_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                on_send_failure(exc)
                return
            deferred.addCallbacks(callback, on_send_failure)

        attempt()

    pagination: dict[str, Any] = {
        "windows": [],
        "window_index": 0,
        "cursor_to_ms": None,
        "trendbars": [],
        "pages_requested": 0,
        "duplicate_bars_removed": 0,
        "has_more_flag_available": False,
        "requested_start_ms": None,
        "requested_end_ms": None,
        "symbol_id": None,
    }

    def finish_ingestion() -> None:
        if failure_state:
            stop_loop()
            return
        try:
            start_ms = int(pagination["requested_start_ms"])
            end_ms = int(pagination["requested_end_ms"])
            unique_bars: dict[int, Any] = {}
            for bar in pagination["trendbars"]:
                timestamp_ms = _trendbar_timestamp_ms(bar)
                if timestamp_ms < start_ms or timestamp_ms > end_ms:
                    continue
                if timestamp_ms in unique_bars:
                    pagination["duplicate_bars_removed"] += 1
                    continue
                unique_bars[timestamp_ms] = bar
            if not unique_bars:
                raise ValueError("cTrader returned no trendbars for this symbol/timeframe/window")
            ordered_bars = [unique_bars[timestamp] for timestamp in sorted(unique_bars)]
            snapshot = normalize_ctrader_trendbars(
                ordered_bars,
                symbol=instrument,
                timeframe=period,
                output_csv=output_csv,
            )
            retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            snapshot["retrieved_at_utc"] = retrieved_at
            snapshot["host"] = "demo"
            snapshot["request_lookback_days"] = days
            snapshot["requested_start_utc"] = datetime.fromtimestamp(start_ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
            snapshot["requested_end_utc"] = datetime.fromtimestamp(end_ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
            snapshot["pagination"] = {
                "complete": True,
                "strategy": "period_limited_time_windows_with_count_pages",
                "window_count": len(pagination["windows"]),
                "pages_requested": pagination["pages_requested"],
                "page_size": _TREND_BAR_PAGE_SIZE,
                "duplicate_bars_removed": pagination["duplicate_bars_removed"],
                "has_more_flag_available": pagination["has_more_flag_available"],
            }
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

    def request_current_page() -> None:
        if failure_state:
            return
        if pagination["pages_requested"] >= _TREND_BAR_MAX_PAGES:
            failure_state["error_type"] = "PaginationPageLimitExceeded"
            stop_loop()
            return
        window_start, window_end = pagination["windows"][pagination["window_index"]]
        cursor_to = pagination["cursor_to_ms"]
        if not isinstance(cursor_to, int) or cursor_to < window_start:
            advance_window()
            return
        request = sdk["ProtoOAGetTrendbarsReq"]()
        request.ctidTraderAccountId = selected_config.account_id
        request.symbolId = int(pagination["symbol_id"])
        request.period = getattr(sdk["ProtoOATrendbarPeriod"], period)
        request.fromTimestamp = int(window_start)
        request.toTimestamp = int(cursor_to)
        request.count = _TREND_BAR_PAGE_SIZE
        pagination["pages_requested"] += 1
        send_with_retry(request, on_bars)

    def advance_window() -> None:
        if failure_state:
            stop_loop()
            return
        next_index = pagination["window_index"] + 1
        if next_index >= len(pagination["windows"]):
            finish_ingestion()
            return
        pagination["window_index"] = next_index
        pagination["cursor_to_ms"] = pagination["windows"][next_index][1]
        request_current_page()

    def get_bars(symbol_id: int) -> None:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        pagination["windows"] = _build_trendbar_windows(start_ms, end_ms, period)
        pagination["window_index"] = 0
        pagination["cursor_to_ms"] = pagination["windows"][0][1]
        pagination["symbol_id"] = int(symbol_id)
        pagination["requested_start_ms"] = pagination["windows"][0][0]
        pagination["requested_end_ms"] = pagination["windows"][-1][1]
        request_current_page()

    def on_bars(message: Any) -> Any:
        if failure_state:
            return message
        try:
            response = extract_response(message)
            if response is None:
                return message
            trendbars = list(response.trendbar)
            window_start, _window_end = pagination["windows"][pagination["window_index"]]
            if not trendbars:
                has_more = _response_has_more_trendbars(response, message)
                if has_more is not None:
                    pagination["has_more_flag_available"] = True
                if has_more is True:
                    raise ValueError("cTrader indicated more trendbars but returned an empty page")
                advance_window()
                return message

            pagination["trendbars"].extend(trendbars)
            has_more = _response_has_more_trendbars(response, message)
            if has_more is not None:
                pagination["has_more_flag_available"] = True
            # Version 0.9.2 of the Python SDK does not expose ProtoOAGetTrendbarsRes.hasMore.
            # If unavailable, a full count-sized page is a conservative signal to keep paging.
            needs_more = has_more if has_more is not None else len(trendbars) >= _TREND_BAR_PAGE_SIZE
            if needs_more:
                oldest_timestamp_ms = min(_trendbar_timestamp_ms(bar) for bar in trendbars)
                current_cursor = int(pagination["cursor_to_ms"])
                if oldest_timestamp_ms <= window_start:
                    advance_window()
                else:
                    next_cursor = oldest_timestamp_ms - 1
                    if next_cursor >= current_cursor:
                        raise ValueError("cTrader trendbar pagination did not move backwards")
                    pagination["cursor_to_ms"] = next_cursor
                    request_current_page()
            else:
                advance_window()
        except Exception as exc:
            failure_state["error_type"] = type(exc).__name__
            stop_loop()
        return message

    def on_symbols(message: Any) -> Any:
        if failure_state:
            return message
        try:
            response = extract_response(message)
            if response is None:
                return message
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
        if failure_state:
            return message
        response = extract_response(message)
        if response is None:
            return message
        request = sdk["ProtoOASymbolsListReq"]()
        request.ctidTraderAccountId = selected_config.account_id
        request.includeArchivedSymbols = False
        send_with_retry(request, on_symbols)
        return message

    def on_application_auth(message: Any) -> Any:
        if failure_state:
            return message
        response = extract_response(message)
        if response is None:
            return message
        request = sdk["ProtoOAAccountAuthReq"]()
        request.ctidTraderAccountId = selected_config.account_id
        request.accessToken = selected_config.access_token
        send_with_retry(request, on_account_auth)
        return message

    def connected(connected_client: Any) -> None:
        request = sdk["ProtoOAApplicationAuthReq"]()
        request.clientId = selected_config.client_id
        request.clientSecret = selected_config.client_secret
        send_with_retry(request, on_application_auth)

    def disconnected(_client: Any, reason: Any) -> None:
        if not result and not failure_state:
            failure_state["error_type"] = "DisconnectedBeforeHistoryResponse"
        stop_loop()

    def on_message(_client: Any, message: Any) -> None:
        # The SDK also surfaces proxy-level ProtoErrorRes messages here; those may not
        # correlate to a request Deferred, so capture their safe code before a timeout hides it.
        try:
            response = sdk["Protobuf"].extract(message)
        except Exception:
            return None
        diagnostic = _safe_api_error_diagnostic(response)
        if diagnostic and not failure_state:
            failure_state["error_type"] = diagnostic
            stop_loop()
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
        message = str(exc)
        # This prefix is constructed internally from a sanitized class/code diagnostic.
        # Do not print arbitrary exception text, which could contain credential-bearing state.
        safe_runtime_prefix = "cTrader research history download failed (diagnostic type: "
        if isinstance(exc, RuntimeError) and (
            message.startswith(safe_runtime_prefix)
            or message == "cTrader research history download ended without a result"
        ):
            diagnostic = message
        else:
            diagnostic = type(exc).__name__
        print(f"cTrader research ingestion failed: {diagnostic}", file=sys.stderr)
        return 2
    pagination = result.get("pagination")
    print(json.dumps({
        "scope": result["scope"],
        "symbol": result["symbol"],
        "timeframe": result["timeframe"],
        "rows": result["rows"],
        "first_timestamp_utc": result.get("first_timestamp_utc"),
        "last_timestamp_utc": result.get("last_timestamp_utc"),
        "requested_start_utc": result.get("requested_start_utc"),
        "requested_end_utc": result.get("requested_end_utc"),
        "snapshot_sha256": result["snapshot_sha256"],
        "host": result["host"],
        "pagination": pagination,
        "actionable": False,
        "output_csv": args.output_csv,
        "manifest_path": str(Path(args.output_csv).with_suffix(Path(args.output_csv).suffix + ".manifest.json")),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
