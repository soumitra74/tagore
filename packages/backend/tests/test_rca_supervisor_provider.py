from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import Settings
from app.providers.rca_supervisor_provider import RcaSupervisorProvider


@pytest.mark.asyncio
async def test_send_calls_messages_create_with_model_and_tools(monkeypatch):
    settings = Settings(
        anthropic_api_key="test-key", rca_supervisor_model="claude-sonnet-5",
        rca_supervisor_temperature=0.7,
    )
    provider = RcaSupervisorProvider(settings)
    fake_response = MagicMock()
    provider._client.messages.create = AsyncMock(return_value=fake_response)

    messages = [{"role": "user", "content": "hello"}]
    tools = [{"name": "noop", "description": "", "input_schema": {"type": "object", "properties": {}}}]
    result = await provider.send(messages, tools, system="You are a helper.")

    assert result is fake_response
    provider._client.messages.create.assert_awaited_once_with(
        model="claude-sonnet-5",
        max_tokens=2048,
        temperature=0.7,
        thinking={"type": "disabled"},
        system="You are a helper.",
        messages=messages,
        tools=tools,
    )
