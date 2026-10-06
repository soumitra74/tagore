from typing import Protocol

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from config import Settings


class CompletionProvider(Protocol):
    async def complete(self, prompt: str) -> str: ...


class AnthropicCompletionProvider:
    def __init__(self, settings: Settings):
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.eval_judge_model

    async def complete(self, prompt: str) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAICompletionProvider:
    def __init__(self, settings: Settings):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.eval_judge_model

    async def complete(self, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class OllamaCompletionProvider:
    def __init__(self, settings: Settings):
        self._base_url = settings.ollama_base_url
        self._model = settings.eval_judge_model

    async def complete(self, prompt: str) -> str:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=120) as client:
            response = await client.post(
                "/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json()["response"]


def get_completion_provider(settings: Settings) -> CompletionProvider:
    if settings.eval_judge_provider == "anthropic":
        return AnthropicCompletionProvider(settings)
    if settings.eval_judge_provider == "openai":
        return OpenAICompletionProvider(settings)
    if settings.eval_judge_provider == "ollama":
        return OllamaCompletionProvider(settings)
    raise ValueError(f"Unknown EVAL_JUDGE_PROVIDER: {settings.eval_judge_provider!r}")
