"""Discover cTrader accounts granted to a token, using the DEMO endpoint only.

This diagnostic performs application authentication and an account-list request. It
never attempts account authentication, requests market data, or sends trading messages.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Sequence

from app.ctrader_history_ingest import (
    CTraderApplicationConfig,
    _safe_api_error_diagnostic,
    _safe_failure,
    load_application_config,
)


def _account_environment(account: Any) -> tuple[str, bool]:
    """Return (environment, is_demo_eligible), treating a missing isLive field as unknown."""
    has_live_flag = False
    is_live = False
    has_field = getattr(account, "HasField", None)
    if callable(has_field):
        try:
            has_live_flag = bool(has_field("isLive"))
            is_live = bool(getattr(account, "isLive", False)) if has_live_flag else False
        except (ValueError, TypeError):
            has_live_flag = False
    elif hasattr(account, "isLive"):
        has_live_flag = True
        is_live = bool(getattr(account, "isLive"))
    if not has_live_flag:
        return "unknown", False
    return ("live", False) if is_live else ("demo", True)


def discover_authorized_accounts(
    *,
    config: CTraderApplicationConfig | None = None,
    client_factory: Callable[..., Any] | None = None,
    reactor_instance: Any | None = None,
    sdk: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List token-granted accounts and mark which IDs are explicitly DEMO accounts."""
    selected_config = config or load_application_config()
    if selected_config.host != "demo":
        raise ValueError("Account discovery is restricted to the cTrader DEMO endpoint")

    if sdk is None:
        from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAApplicationAuthReq,
            ProtoOAGetAccountListByAccessTokenReq,
        )
        from twisted.internet import reactor as twisted_reactor

        sdk = {
            "Client": Client,
            "Protobuf": Protobuf,
            "TcpProtocol": TcpProtocol,
            "EndPoints": EndPoints,
            "ProtoOAApplicationAuthReq": ProtoOAApplicationAuthReq,
            "ProtoOAGetAccountListByAccessTokenReq": ProtoOAGetAccountListByAccessTokenReq,
            "reactor": twisted_reactor,
        }

    loop = reactor_instance or sdk["reactor"]
    factory = client_factory or sdk["Client"]
    client = factory(
        sdk["EndPoints"].PROTOBUF_DEMO_HOST,
        sdk["EndPoints"].PROTOBUF_PORT,
        sdk["TcpProtocol"],
    )
    state: dict[str, Any] = {"accounts": None}
    failures: dict[str, str] = {}

    def stop_loop() -> None:
        if getattr(loop, "running", False):
            loop.stop()

    def fail(failure: Any) -> Any:
        failures["diagnostic"] = _safe_failure(failure)
        stop_loop()
        return failure

    def extract_response(message: Any) -> Any | None:
        try:
            response = sdk["Protobuf"].extract(message)
        except Exception as exc:
            failures["diagnostic"] = type(exc).__name__
            stop_loop()
            return None
        diagnostic = _safe_api_error_diagnostic(response)
        if diagnostic:
            failures["diagnostic"] = diagnostic
            stop_loop()
            return None
        return response

    def on_account_list(message: Any) -> Any:
        if failures:
            return message
        try:
            response = extract_response(message)
            if response is None:
                return message
            discovered = []
            for account in getattr(response, "ctidTraderAccount", []):
                try:
                    account_id = int(getattr(account, "ctidTraderAccountId"))
                except (TypeError, ValueError, AttributeError):
                    continue
                if account_id <= 0:
                    continue
                environment, demo_eligible = _account_environment(account)
                broker = str(getattr(account, "brokerTitleShort", "") or "").strip()
                discovered.append({
                    "account_id": account_id,
                    "environment": environment,
                    "demo_eligible": demo_eligible,
                    "broker": broker or None,
                })
            discovered.sort(key=lambda item: (item["environment"], item["account_id"]))
            state["accounts"] = discovered
        except Exception as exc:
            failures["diagnostic"] = type(exc).__name__
        finally:
            stop_loop()
        return message

    def on_application_auth(message: Any) -> Any:
        if failures:
            return message
        response = extract_response(message)
        if response is None:
            return message
        request = sdk["ProtoOAGetAccountListByAccessTokenReq"]()
        request.accessToken = selected_config.access_token
        client.send(request).addCallbacks(on_account_list, fail)
        return message

    def connected(connected_client: Any) -> None:
        request = sdk["ProtoOAApplicationAuthReq"]()
        request.clientId = selected_config.client_id
        request.clientSecret = selected_config.client_secret
        connected_client.send(request).addCallbacks(on_application_auth, fail)

    def disconnected(_client: Any, _reason: Any) -> None:
        if state["accounts"] is None and not failures:
            failures["diagnostic"] = "DisconnectedBeforeAccountListResponse"
        stop_loop()

    def on_message(_client: Any, message: Any) -> None:
        # Capture proxy-level error responses even if they do not resolve a request Deferred.
        try:
            response = sdk["Protobuf"].extract(message)
        except Exception:
            return None
        diagnostic = _safe_api_error_diagnostic(response)
        if diagnostic and not failures:
            failures["diagnostic"] = diagnostic
            stop_loop()
        return None

    client.setConnectedCallback(connected)
    client.setDisconnectedCallback(disconnected)
    client.setMessageReceivedCallback(on_message)
    client.startService()
    loop.run()

    if failures:
        raise RuntimeError(
            "cTrader account discovery failed (diagnostic type: "
            + failures["diagnostic"] + ")"
        )
    if state["accounts"] is None:
        raise RuntimeError("cTrader account discovery ended without an account-list response")

    accounts = state["accounts"]
    demo_accounts = [account for account in accounts if account["demo_eligible"]]
    return {
        "scope": "CTRADER_OPEN_API_ACCOUNT_DISCOVERY",
        "status": "PASS" if demo_accounts else "NO_DEMO_ACCOUNTS_FOUND",
        "host": "demo",
        "total_accounts_returned": len(accounts),
        "demo_account_count": len(demo_accounts),
        "accounts": accounts,
        "actionable": False,
        "execution_connected": False,
        "network_checked": True,
        "credential_values_exposed": False,
        "orders_possible_from_this_check": False,
        "note": (
            "Use an account_id marked demo_eligible=true in CTRADER_ACCOUNT_ID. "
            "This diagnostic does not authorize the account or request market data."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List cTrader token-granted accounts; use DEMO endpoint only."
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)
    try:
        result = discover_authorized_accounts()
    except Exception as exc:
        message = str(exc)
        prefix = "cTrader account discovery failed (diagnostic type: "
        if isinstance(exc, RuntimeError) and message.startswith(prefix):
            diagnostic = message
        else:
            diagnostic = type(exc).__name__
        print(f"cTrader account discovery failed: {diagnostic}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("cTrader DEMO account discovery")
        print(f"Accounts returned: {result['total_accounts_returned']}")
        print(f"DEMO accounts: {result['demo_account_count']}")
        for account in result["accounts"]:
            broker = f" | broker={account['broker']}" if account["broker"] else ""
            print(
                f"account_id={account['account_id']} | environment={account['environment']} "
                f"| demo_eligible={str(account['demo_eligible']).lower()}{broker}"
            )
        if not result["demo_account_count"]:
            print("No account explicitly identified as DEMO; do not guess an account ID.")
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
