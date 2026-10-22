import asyncio

import httpx
import pytest

from app.config import get_settings
from app.llama_client import InferenceUnavailable, LlamaClient


class FakeResponse:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://127.0.0.1:8080/v1/completions")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("HTTP error", request=request, response=response)

    def json(self):
        return self.body


class FakeAsyncClient:
    calls = []
    models_body = {
        "data": [{
            "id": "/home/jarvis/llama-models/qwen.gguf",
            "owned_by": "llamacpp",
        }]
    }
    completion_body = {
        "object": "text_completion",
        "choices": [{
            "text": '{"decision":"NO_TRADE","market_regime":"UNCERTAIN","reason_codes":["INSUFFICIENT_EVIDENCE"]}'
        }],
    }
    completion_status = 200

    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url):
        self.calls.append(("GET", url, None))
        return FakeResponse(self.models_body)

    async def post(self, url, json):
        self.calls.append(("POST", url, json))
        return FakeResponse(self.completion_body, self.completion_status)


def _settings(**overrides):
    values = {
        "llama_model_id": "",
        "llama_base_url": "http://127.0.0.1:8080",
        "llama_timeout_seconds": 2,
        "llama_max_tokens": 64,
    }
    values.update(overrides)
    return get_settings().model_copy(update=values)


@pytest.fixture(autouse=True)
def fake_httpx(monkeypatch):
    FakeAsyncClient.calls = []
    FakeAsyncClient.models_body = {
        "data": [{
            "id": "/home/jarvis/llama-models/qwen.gguf",
            "owned_by": "llamacpp",
        }]
    }
    FakeAsyncClient.completion_body = {
        "object": "text_completion",
        "choices": [{
            "text": '{"decision":"NO_TRADE","market_regime":"UNCERTAIN","reason_codes":["INSUFFICIENT_EVIDENCE"]}'
        }],
    }
    FakeAsyncClient.completion_status = 200
    monkeypatch.setattr("app.llama_client.httpx.AsyncClient", FakeAsyncClient)


def test_llama_client_uses_completion_endpoint_and_model_id():
    model_id, content = asyncio.run(
        LlamaClient(_settings()).analyze({"readiness_probe": True})
    )

    assert model_id == "/home/jarvis/llama-models/qwen.gguf"
    assert '"decision":"NO_TRADE"' in content
    assert ("GET", "http://127.0.0.1:8080/v1/models", None) in FakeAsyncClient.calls
    post_calls = [call for call in FakeAsyncClient.calls if call[0] == "POST"]
    assert len(post_calls) == 1
    method, url, payload = post_calls[0]
    assert method == "POST"
    assert url == "http://127.0.0.1:8080/v1/completions"
    assert payload["model"] == model_id
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 64
    assert "prompt" in payload
    assert "messages" not in payload


def test_llama_client_uses_explicit_model_id_when_configured():
    model_id, _ = asyncio.run(
        LlamaClient(_settings(llama_model_id="/home/jarvis/llama-models/qwen.gguf")).analyze(
            {"readiness_probe": True}
        )
    )
    assert model_id == "/home/jarvis/llama-models/qwen.gguf"


def test_llama_client_rejects_ambiguous_model_listing():
    FakeAsyncClient.models_body = {"data": [{"id": "model-a"}, {"id": "model-b"}]}

    with pytest.raises(InferenceUnavailable, match="Configured model ID"):
        asyncio.run(LlamaClient(_settings()).analyze({"readiness_probe": True}))

    assert not any(call[0] == "POST" for call in FakeAsyncClient.calls)


def test_llama_client_rejects_configured_model_not_in_listing():
    with pytest.raises(InferenceUnavailable, match="Configured model ID"):
        asyncio.run(
            LlamaClient(_settings(llama_model_id="not-loaded")).analyze({"readiness_probe": True})
        )


def test_llama_client_rejects_missing_completion_text():
    FakeAsyncClient.completion_body = {"choices": [{"message": {"content": "not text completion"}}]}

    with pytest.raises(InferenceUnavailable, match="inference failed"):
        asyncio.run(LlamaClient(_settings()).analyze({"readiness_probe": True}))


def test_llama_client_converts_http_error_to_safe_inference_error():
    FakeAsyncClient.completion_status = 503

    with pytest.raises(InferenceUnavailable, match="inference failed"):
        asyncio.run(LlamaClient(_settings()).analyze({"readiness_probe": True}))
