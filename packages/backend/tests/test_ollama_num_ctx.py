import json

import httpx
import pytest

from app.config import Settings
from app.providers.ollama_provider import OllamaChatProvider
from app.services.judge import OllamaCompletionProvider


def _mock_transport(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"response": "ok", "prompt_eval_count": 1, "eval_count": 1})

    return httpx.MockTransport(handler)


_RealAsyncClient = httpx.AsyncClient


@pytest.mark.asyncio
async def test_chat_provider_sends_configured_num_ctx(monkeypatch):
    settings = Settings(ollama_num_ctx=4096)
    provider = OllamaChatProvider(settings)
    captured: dict = {}
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **kw: _RealAsyncClient(*a, transport=_mock_transport(captured), **kw),
    )
    await provider.generate("q", ["ctx"])
    assert captured["body"]["options"]["num_ctx"] == 4096


@pytest.mark.asyncio
async def test_judge_completion_provider_sends_configured_num_ctx(monkeypatch):
    settings = Settings(ollama_num_ctx=4096, eval_judge_temperature=0.0)
    provider = OllamaCompletionProvider(settings)
    captured: dict = {}
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **kw: _RealAsyncClient(*a, transport=_mock_transport(captured), **kw),
    )
    await provider.complete("prompt")
    assert captured["body"]["options"] == {"temperature": 0.0, "num_ctx": 4096}


def test_default_num_ctx_is_generous():
    assert Settings().ollama_num_ctx >= 4096
