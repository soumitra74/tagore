from anthropic import AsyncAnthropic

from app.config import Settings


class RcaSupervisorProvider:
    """Thin wrapper around Anthropic's native tool_use support. Distinct from
    AnthropicChatProvider (providers/anthropic_provider.py), which is a
    single-shot generate(question, context_chunks) call with no tool-calling
    -- the RCA supervisor needs a multi-turn tool_use/tool_result loop
    instead, driven by rca_agent.run_turn."""

    def __init__(self, settings: Settings):
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=300)
        self._model = settings.rca_supervisor_model
        self._temperature = settings.rca_supervisor_temperature

    async def send(self, messages: list[dict], tools: list[dict], system: str):
        return await self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            temperature=self._temperature,
            thinking={"type": "disabled"},
            system=system,
            messages=messages,
            tools=tools,
        )
