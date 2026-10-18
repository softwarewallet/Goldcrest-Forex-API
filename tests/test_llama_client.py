import asyncio

from app.config import get_settings
from app.llama_client import LlamaClient


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class FakeAsyncClient:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url):
        self.calls.append(("GET", url, None))
        return FakeResponse({
            "data": [{
                "id": "/home/jarvis/llama-models/qwen.gguf",
                "owned_by": "llamacpp"
            }]
        })

    async def post(self, url, json):
        self.calls.append(("POST", url, json))
        return FakeResponse({
            "object": "text_completion",
            "choices": [{"text": '{"decision":"NO_TRADE","market_regime":"UNCERTAIN","reason_codes":["INSUFFICIENT_EVIDENCE"]}'}]
        })


def test_llama_client_uses_completion_endpoint_and_model_id(monkeypatch):
    import app.llama_client as module

    FakeAsyncClient.calls = []
    monkeypatch.setattr(module.httpx, "AsyncClient", FakeAsyncClient)
    settings = get_settings().model_copy(update={
        "llama_model_id": "",
        "llama_base_url": "http://127.0.0.1:8080",
    })

    model_id, content = asyncio.run(LlamaClient(settings).analyze({"readiness_probe": True}))

    assert model_id == "/home/jarvis/llama-models/qwen.gguf"
    assert '"decision":"NO_TRADE"' in content
    post_calls = [call for call in FakeAsyncClient.calls if call[0] == "POST"]
    assert len(post_calls) == 1
    assert post_calls[0][1] == "http://127.0.0.1:8080/v1/completions"
    assert post_calls[0][2]["model"] == "/home/jarvis/llama-models/qwen.gguf"
    assert "prompt" in post_calls[0][2]
    assert "messages" not in post_calls[0][2]
