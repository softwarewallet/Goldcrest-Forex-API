from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import ctrader_history_ingest
from app.ctrader_history_ingest import (
    CTraderResearchConfig,
    download_ctrader_history,
    load_config,
    validate_request,
)


def test_config_reads_credentials_without_echoing_and_defaults_to_demo():
    config = load_config({
        "CTRADER_CLIENT_ID": "client-id",
        "CTRADER_CLIENT_SECRET": "secret-value",
        "CTRADER_ACCESS_TOKEN": "access-value",
        "CTRADER_ACCOUNT_ID": "12345",
    })
    assert config.host == "demo"
    assert config.account_id == 12345
    assert config.client_secret == "secret-value"


def test_default_config_loads_credentials_from_repository_root_dotenv(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "CTRADER_HOST=demo\n"
        "CTRADER_CLIENT_ID=file-client-id\n"
        "CTRADER_CLIENT_SECRET='file-client-secret'\n"
        "CTRADER_ACCESS_TOKEN=file-access-token\n"
        "CTRADER_ACCOUNT_ID=123456789\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ctrader_history_ingest, "_PROJECT_ROOT", tmp_path)
    for name in (
        "CTRADER_HOST", "CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET",
        "CTRADER_ACCESS_TOKEN", "CTRADER_ACCOUNT_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_config()

    assert config.host == "demo"
    assert config.client_id == "file-client-id"
    assert config.client_secret == "file-client-secret"
    assert config.access_token == "file-access-token"
    assert config.account_id == 123456789


def test_process_environment_takes_precedence_over_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "CTRADER_CLIENT_ID=file-client-id\n"
        "CTRADER_CLIENT_SECRET=file-client-secret\n"
        "CTRADER_ACCESS_TOKEN=file-access-token\n"
        "CTRADER_ACCOUNT_ID=123456789\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ctrader_history_ingest, "_PROJECT_ROOT", tmp_path)
    for name in (
        "CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET",
        "CTRADER_ACCESS_TOKEN", "CTRADER_ACCOUNT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CTRADER_CLIENT_ID", "shell-client-id")

    config = load_config()

    assert config.client_id == "shell-client-id"
    assert config.client_secret == "file-client-secret"


def test_empty_process_environment_values_do_not_shadow_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "CTRADER_HOST=demo\n"
        "CTRADER_CLIENT_ID=file-client-id\n"
        "CTRADER_CLIENT_SECRET=file-client-secret\n"
        "CTRADER_ACCESS_TOKEN=file-access-token\n"
        "CTRADER_ACCOUNT_ID=123456789\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ctrader_history_ingest, "_PROJECT_ROOT", tmp_path)
    # Some shells/service managers export empty keys; those must not block the .env.
    for name in (
        "CTRADER_HOST", "CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET",
        "CTRADER_ACCESS_TOKEN", "CTRADER_ACCOUNT_ID",
    ):
        monkeypatch.setenv(name, "")

    config = load_config()

    assert config.host == "demo"
    assert config.client_id == "file-client-id"
    assert config.client_secret == "file-client-secret"
    assert config.access_token == "file-access-token"
    assert config.account_id == 123456789


def test_api_error_diagnostic_exposes_error_code_but_not_description():
    response = type("ProtoOAErrorRes", (), {
        "errorCode": "OA_AUTH_TOKEN_EXPIRED",
        "description": "do-not-log-token-or-other-sensitive-description",
    })()

    diagnostic = ctrader_history_ingest._safe_api_error_diagnostic(response)

    assert diagnostic == "ProtoOAErrorRes code=OA_AUTH_TOKEN_EXPIRED"
    assert "do-not-log" not in diagnostic


def test_cli_prints_safe_download_diagnostic_and_hides_arbitrary_exception_text(monkeypatch, capsys):
    def fail_with_safe_diagnostic(**_kwargs):
        raise RuntimeError(
            "cTrader research history download failed "
            "(diagnostic type: ProtoOAErrorRes code=OA_AUTH_TOKEN_EXPIRED)"
        )

    monkeypatch.setattr(ctrader_history_ingest, "download_ctrader_history", fail_with_safe_diagnostic)
    exit_code = ctrader_history_ingest.main([
        "--symbol", "EURUSD", "--timeframe", "M15", "--lookback-days", "7",
        "--output-csv", "unused.csv",
    ])
    assert exit_code == 2
    assert "ProtoOAErrorRes code=OA_AUTH_TOKEN_EXPIRED" in capsys.readouterr().err

    def fail_with_untrusted_message(**_kwargs):
        raise RuntimeError("secret-token-must-not-be-shown")

    monkeypatch.setattr(ctrader_history_ingest, "download_ctrader_history", fail_with_untrusted_message)
    exit_code = ctrader_history_ingest.main([
        "--symbol", "EURUSD", "--timeframe", "M15", "--lookback-days", "7",
        "--output-csv", "unused.csv",
    ])
    assert exit_code == 2
    output = capsys.readouterr().err
    assert "RuntimeError" in output
    assert "secret-token-must-not-be-shown" not in output


def test_application_config_does_not_require_account_id():
    from app.ctrader_history_ingest import load_application_config

    config = load_application_config({
        "CTRADER_CLIENT_ID": "client-id",
        "CTRADER_CLIENT_SECRET": "secret-value",
        "CTRADER_ACCESS_TOKEN": "access-value",
        "CTRADER_HOST": "demo",
    })

    assert config.host == "demo"
    assert config.client_id == "client-id"
    assert config.access_token == "access-value"


def test_account_discovery_reports_only_accounts_explicitly_marked_demo():
    from app.ctrader_account_discovery import discover_authorized_accounts, _account_environment

    class FakeGetAccountListByAccessTokenReq(FakeRequest):
        pass

    accounts = [
        SimpleNamespace(ctidTraderAccountId=987654, isLive=False, brokerTitleShort="DemoBroker"),
        SimpleNamespace(ctidTraderAccountId=123456, isLive=True, brokerTitleShort="LiveBroker"),
        SimpleNamespace(ctidTraderAccountId=555555, brokerTitleShort="UnknownFlagBroker"),
    ]
    responses = {
        "ProtoOAApplicationAuthReq": object(),
        "FakeGetAccountListByAccessTokenReq": SimpleNamespace(ctidTraderAccount=accounts),
    }
    sdk = {
        "Client": lambda host, port, protocol: FakeClient(host, port, protocol, responses),
        "Protobuf": SimpleNamespace(extract=lambda value: value),
        "TcpProtocol": object(),
        "EndPoints": SimpleNamespace(PROTOBUF_DEMO_HOST="demo-host", PROTOBUF_PORT=5035),
        "ProtoOAApplicationAuthReq": type("ProtoOAApplicationAuthReq", (FakeRequest,), {}),
        "ProtoOAGetAccountListByAccessTokenReq": FakeGetAccountListByAccessTokenReq,
        "reactor": FakeReactor(),
    }
    config = ctrader_history_ingest.load_application_config({
        "CTRADER_CLIENT_ID": "id", "CTRADER_CLIENT_SECRET": "secret",
        "CTRADER_ACCESS_TOKEN": "token", "CTRADER_HOST": "demo",
    })

    report = discover_authorized_accounts(config=config, sdk=sdk)

    assert report["status"] == "PASS"
    assert report["total_accounts_returned"] == 3
    assert report["demo_account_count"] == 1
    by_id = {account["account_id"]: account for account in report["accounts"]}
    assert by_id[987654]["demo_eligible"] is True
    assert by_id[123456]["demo_eligible"] is False
    assert by_id[123456]["environment"] == "live"
    assert by_id[555555]["environment"] == "unknown"
    assert by_id[555555]["demo_eligible"] is False
    assert report["credential_values_exposed"] is False
    assert report["orders_possible_from_this_check"] is False


def test_account_discovery_fails_closed_when_live_host_is_requested():
    from app.ctrader_account_discovery import discover_authorized_accounts
    from app.ctrader_history_ingest import CTraderApplicationConfig

    config = CTraderApplicationConfig("id", "secret", "token", host="live")
    with pytest.raises(ValueError, match="DEMO endpoint"):
        discover_authorized_accounts(config=config)


def test_config_fails_closed_for_live_host_or_missing_credentials():
    with pytest.raises(ValueError, match="Missing required"):
        load_config({})
    env = {
        "CTRADER_CLIENT_ID": "id",
        "CTRADER_CLIENT_SECRET": "secret",
        "CTRADER_ACCESS_TOKEN": "token",
        "CTRADER_ACCOUNT_ID": "12345",
        "CTRADER_HOST": "live",
    }
    with pytest.raises(ValueError, match="restricted to DEMO"):
        load_config(env)


@pytest.mark.parametrize(
    "symbol,timeframe,days",
    [("", "M15", 7), ("EURUSD", "M2", 7), ("EURUSD", "M15", 0), ("EURUSD", "M15", 3651)],
)
def test_request_validation_rejects_invalid_parameters(symbol, timeframe, days):
    with pytest.raises(ValueError):
        validate_request(symbol, timeframe, days)


class FakeDeferred:
    def __init__(self, value):
        self.value = value

    def addCallbacks(self, callback, errback):
        try:
            callback(self.value)
        except Exception as exc:
            errback(exc)
        return self


class FakeRequest:
    pass


class FakeClient:
    def __init__(self, host, port, protocol, responses):
        self.host, self.port, self.protocol = host, port, protocol
        self.responses = responses
        self.sent = []
        self.connected_callback = None

    def setConnectedCallback(self, callback):
        self.connected_callback = callback

    def setDisconnectedCallback(self, callback):
        self.disconnected_callback = callback

    def setMessageReceivedCallback(self, callback):
        self.message_callback = callback

    def startService(self):
        self.connected_callback(self)

    def send(self, request):
        self.sent.append(request)
        name = type(request).__name__
        return FakeDeferred(self.responses[name])


class FakeReactor:
    running = False

    def run(self):
        return None

    def stop(self):
        return None

    def callLater(self, _delay, callback, *args, **kwargs):
        return callback(*args, **kwargs)


def test_trendbar_windows_are_minute_aligned_bounded_and_overlap_without_gaps():
    minute_ms = 60_000
    start_ms = 1_700_000_123_456
    end_ms = start_ms + 800 * 24 * 60 * minute_ms

    windows = ctrader_history_ingest._build_trendbar_windows(start_ms, end_ms, "M15")

    assert len(windows) > 1
    assert windows[0][0] == (start_ms // minute_ms) * minute_ms
    assert windows[-1][1] == (end_ms // minute_ms) * minute_ms
    for window_start, window_end in windows:
        assert window_start % minute_ms == 0
        assert window_end % minute_ms == 0
        assert window_end - window_start <= 240 * 24 * 60 * minute_ms
    for previous, current in zip(windows, windows[1:]):
        assert previous[1] == current[0]


def test_optional_has_more_flag_falls_back_when_sdk_does_not_expose_it():
    assert ctrader_history_ingest._response_has_more_trendbars(SimpleNamespace(trendbar=[])) is None
    assert ctrader_history_ingest._response_has_more_trendbars(SimpleNamespace(hasMore=True)) is True
    assert ctrader_history_ingest._response_has_more_trendbars(SimpleNamespace(hasMore=False)) is False


def test_has_more_is_read_from_raw_protobuf_when_old_sdk_schema_omits_field():
    response = SimpleNamespace(trendbar=[])
    # field 1 (varint), field 5 (length-delimited), then field 7 (optional bool)
    payload_with_more = bytes((0x08, 0x02, 0x2A, 0x03)) + b"abc" + bytes((0x38, 0x01))
    payload_without_more = bytes((0x38, 0x00))

    assert ctrader_history_ingest._response_has_more_trendbars(
        response, SimpleNamespace(payload=payload_with_more)
    ) is True
    assert ctrader_history_ingest._response_has_more_trendbars(
        response, SimpleNamespace(payload=payload_without_more)
    ) is False


def test_read_only_download_runs_auth_symbol_and_trendbar_flow(tmp_path):
    from app.ctrader_history_ingest import download_ctrader_history

    latest_minute = int(datetime.now(timezone.utc).timestamp() // 60)
    bars = [
        SimpleNamespace(utcTimestampInMinutes=latest_minute - 15, low=110000, deltaOpen=200, deltaHigh=800, deltaClose=500),
        SimpleNamespace(utcTimestampInMinutes=latest_minute, low=110100, deltaOpen=100, deltaHigh=600, deltaClose=400),
    ]
    responses = {
        "ProtoOAApplicationAuthReq": object(),
        "ProtoOAAccountAuthReq": object(),
        "ProtoOASymbolsListReq": SimpleNamespace(symbol=[
            SimpleNamespace(symbolName="EURUSD", symbolId=1),
        ]),
        "ProtoOAGetTrendbarsReq": SimpleNamespace(trendbar=bars),
    }
    sdk = {
        "Client": lambda host, port, protocol: FakeClient(host, port, protocol, responses),
        "Protobuf": SimpleNamespace(extract=lambda value: value),
        "TcpProtocol": object(),
        "EndPoints": SimpleNamespace(PROTOBUF_DEMO_HOST="demo-host", PROTOBUF_PORT=5035),
        "ProtoOAApplicationAuthReq": type("ProtoOAApplicationAuthReq", (FakeRequest,), {}),
        "ProtoOAAccountAuthReq": type("ProtoOAAccountAuthReq", (FakeRequest,), {}),
        "ProtoOASymbolsListReq": type("ProtoOASymbolsListReq", (FakeRequest,), {}),
        "ProtoOAGetTrendbarsReq": type("ProtoOAGetTrendbarsReq", (FakeRequest,), {}),
        "ProtoOATrendbarPeriod": SimpleNamespace(M15=15),
        "reactor": FakeReactor(),
    }
    output = tmp_path / "EURUSD_M15.csv"
    config = CTraderResearchConfig("id", "secret", "token", 12345, "demo")
    result = download_ctrader_history(
        symbol="EURUSD",
        timeframe="M15",
        lookback_days=30,
        output_csv=output,
        config=config,
        sdk=sdk,
    )
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert result["actionable"] is False
    assert result["host"] == "demo"
    assert result["rows"] == 2
    assert result["snapshot_sha256"]
    assert result["pagination"]["complete"] is True
    assert result["pagination"]["pages_requested"] == 1
    assert result["pagination"]["window_count"] == 1
    assert Path(str(output) + ".manifest.json").exists()



def test_history_download_paginates_with_older_sdk_count_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(ctrader_history_ingest, "_TREND_BAR_PAGE_SIZE", 2)
    latest_minute = int(datetime.now(timezone.utc).timestamp() // 60)

    def bar(minute):
        return SimpleNamespace(
            utcTimestampInMinutes=minute,
            low=110000,
            deltaOpen=100,
            deltaHigh=500,
            deltaClose=300,
        )

    # ctrader-open-api==0.9.2 doesn't expose ProtoOAGetTrendbarsRes.hasMore.
    # An exactly full count page must therefore trigger a backwards request.
    newest_page = SimpleNamespace(trendbar=[bar(latest_minute - 15), bar(latest_minute)])
    older_page = SimpleNamespace(trendbar=[bar(latest_minute - 30)])

    class PagedFakeClient(FakeClient):
        def send(self, request):
            self.sent.append(request)
            name = type(request).__name__
            if name == "ProtoOAGetTrendbarsReq":
                response = self.responses[name].pop(0)
            else:
                response = self.responses[name]
            return FakeDeferred(response)

    responses = {
        "ProtoOAApplicationAuthReq": object(),
        "ProtoOAAccountAuthReq": object(),
        "ProtoOASymbolsListReq": SimpleNamespace(symbol=[
            SimpleNamespace(symbolName="EURUSD", symbolId=1),
        ]),
        "ProtoOAGetTrendbarsReq": [newest_page, older_page],
    }
    clients = []

    def factory(host, port, protocol):
        client = PagedFakeClient(host, port, protocol, responses)
        clients.append(client)
        return client

    sdk = {
        "Client": factory,
        "Protobuf": SimpleNamespace(extract=lambda value: value),
        "TcpProtocol": object(),
        "EndPoints": SimpleNamespace(PROTOBUF_DEMO_HOST="demo-host", PROTOBUF_PORT=5035),
        "ProtoOAApplicationAuthReq": type("ProtoOAApplicationAuthReq", (FakeRequest,), {}),
        "ProtoOAAccountAuthReq": type("ProtoOAAccountAuthReq", (FakeRequest,), {}),
        "ProtoOASymbolsListReq": type("ProtoOASymbolsListReq", (FakeRequest,), {}),
        "ProtoOAGetTrendbarsReq": type("ProtoOAGetTrendbarsReq", (FakeRequest,), {}),
        "ProtoOATrendbarPeriod": SimpleNamespace(M15=15),
        "reactor": FakeReactor(),
    }
    output = tmp_path / "EURUSD_M15_paged.csv"
    result = download_ctrader_history(
        symbol="EURUSD",
        timeframe="M15",
        lookback_days=7,
        output_csv=output,
        config=CTraderResearchConfig("id", "secret", "token", 12345, "demo"),
        sdk=sdk,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    trendbar_requests = [item for item in clients[0].sent if type(item).__name__ == "ProtoOAGetTrendbarsReq"]
    timestamps = [row["timestamp"] for row in rows]

    assert len(rows) == 3
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == 3
    assert len(trendbar_requests) == 2
    assert all(request.count == 2 for request in trendbar_requests)
    assert trendbar_requests[1].toTimestamp < trendbar_requests[0].toTimestamp
    assert trendbar_requests[1].fromTimestamp == trendbar_requests[0].fromTimestamp
    assert result["pagination"]["complete"] is True
    assert result["pagination"]["pages_requested"] == 2
    assert result["pagination"]["has_more_flag_available"] is False
    assert result["rows"] == 3



def test_transient_ctrader_request_timeout_retries_with_bounded_attempts(tmp_path):
    latest_minute = int(datetime.now(timezone.utc).timestamp() // 60)
    bars = [SimpleNamespace(utcTimestampInMinutes=latest_minute, low=110000,
                            deltaOpen=100, deltaHigh=500, deltaClose=300)]

    class TimeoutDeferred:
        def addCallbacks(self, _callback, errback):
            errback(SimpleNamespace(value=TimeoutError("private detail")))
            return self

    class RetryClient(FakeClient):
        def send(self, request):
            self.sent.append(request)
            name = type(request).__name__
            if name == "ProtoOAApplicationAuthReq" and sum(
                type(item).__name__ == name for item in self.sent
            ) == 1:
                return TimeoutDeferred()
            return FakeDeferred(self.responses[name])

    responses = {
        "ProtoOAApplicationAuthReq": object(),
        "ProtoOAAccountAuthReq": object(),
        "ProtoOASymbolsListReq": SimpleNamespace(symbol=[SimpleNamespace(symbolName="EURUSD", symbolId=1)]),
        "ProtoOAGetTrendbarsReq": SimpleNamespace(trendbar=bars),
    }
    clients = []
    def factory(host, port, protocol):
        client = RetryClient(host, port, protocol, responses)
        clients.append(client)
        return client

    sdk = {
        "Client": factory,
        "Protobuf": SimpleNamespace(extract=lambda value: value),
        "TcpProtocol": object(),
        "EndPoints": SimpleNamespace(PROTOBUF_DEMO_HOST="demo-host", PROTOBUF_PORT=5035),
        "ProtoOAApplicationAuthReq": type("ProtoOAApplicationAuthReq", (FakeRequest,), {}),
        "ProtoOAAccountAuthReq": type("ProtoOAAccountAuthReq", (FakeRequest,), {}),
        "ProtoOASymbolsListReq": type("ProtoOASymbolsListReq", (FakeRequest,), {}),
        "ProtoOAGetTrendbarsReq": type("ProtoOAGetTrendbarsReq", (FakeRequest,), {}),
        "ProtoOATrendbarPeriod": SimpleNamespace(M15=15),
        "reactor": FakeReactor(),
    }
    output = tmp_path / "EURUSD_M15_retry.csv"
    report = download_ctrader_history(
        symbol="EURUSD", timeframe="M15", lookback_days=7, output_csv=output,
        config=CTraderResearchConfig("id", "secret", "token", 12345, "demo"), sdk=sdk,
    )
    auth_requests = [request for request in clients[0].sent if type(request).__name__ == "ProtoOAApplicationAuthReq"]
    assert len(auth_requests) == 2
    assert report["pagination"]["complete"] is True
    assert output.is_file()
