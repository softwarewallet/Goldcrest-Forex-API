from __future__ import annotations

import csv
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


def test_read_only_download_runs_auth_symbol_and_trendbar_flow(tmp_path):
    from app.ctrader_history_ingest import download_ctrader_history

    bars = [
        SimpleNamespace(utcTimestampInMinutes=29_000_000, low=110000, deltaOpen=200, deltaHigh=800, deltaClose=500),
        SimpleNamespace(utcTimestampInMinutes=29_000_015, low=110100, deltaOpen=100, deltaHigh=600, deltaClose=400),
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
    assert Path(str(output) + ".manifest.json").exists()
