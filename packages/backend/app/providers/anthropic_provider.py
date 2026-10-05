from anthropic import AsyncAnthropic

from app.config import Settings
from app.providers.base import LLMResponse, build_prompt


class AnthropicChatProvider:
    def __init__(self, settings: Settings):
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=300)
        self._model = settings.anthropic_model

    async def generate(self, question: str, context_chunks: list[str]) -> LLMResponse:
        prompt = build_prompt(question, context_chunks)
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
