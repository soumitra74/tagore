from openai import AsyncOpenAI

from app.providers.base import LLMResponse
from app.routing.router import Tier


class OpenAICompatibleClient:
    """Calls any OpenAI-compatible endpoint (the gateway, Ollama /v1, vLLM)."""

    def __init__(self, timeout: float = 300):
        self._timeout = timeout
        self._clients: dict[str, AsyncOpenAI] = {}

    def _client_for(self, tier: Tier) -> AsyncOpenAI:
        if tier.name not in self._clients:
            self._clients[tier.name] = AsyncOpenAI(
                base_url=tier.base_url,
                api_key=tier.api_key or "not-needed",
                timeout=self._timeout,
            )
        return self._clients[tier.name]

    async def complete(self, tier: Tier, messages: list[dict]) -> LLMResponse:
        response = await self._client_for(tier).chat.completions.create(
            model=tier.model, messages=messages, temperature=tier.temperature
        )
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
