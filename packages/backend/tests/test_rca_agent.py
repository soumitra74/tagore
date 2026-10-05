from unittest.mock import MagicMock

import pytest

from app.services.rca_agent import run_turn, to_anthropic_messages


def _text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(tool_id, name, input_):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_
    return block


def _response(content, stop_reason, input_tokens=100, output_tokens=50):
    response = MagicMock()
    response.content = content
    response.stop_reason = stop_reason
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


class FakeDatabase:
    async def get_eval_run_detail(self, cycle_id, query_id):
        return {"cycle_id": cycle_id, "query_id": query_id, "question": "serial number?"}


class ScriptedProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def send(self, messages, tools, system):
        self.calls.append([dict(m) for m in messages if isinstance(m, dict)])
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_run_turn_returns_text_when_no_tool_use():
    provider = ScriptedProvider([
        _response([_text_block("No investigation needed.")], "end_turn"),
    ])

    text, tool_trace, input_tokens, output_tokens = await run_turn(
        provider, FakeDatabase(), dim=768, log_file="unused.log",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert text == "No investigation needed."
    assert tool_trace == []
    assert input_tokens == 100
    assert output_tokens == 50


@pytest.mark.asyncio
async def test_run_turn_executes_tool_and_returns_final_text():
    provider = ScriptedProvider([
        _response(
            [_tool_use_block("t1", "get_eval_run", {"cycle_id": 62, "query_id": 1})],
            "tool_use",
        ),
        _response([_text_block("The question is about a serial number.")], "end_turn"),
    ])

    text, tool_trace, input_tokens, output_tokens = await run_turn(
        provider, FakeDatabase(), dim=768, log_file="unused.log",
        messages=[{"role": "user", "content": "Investigate cycle 62, question 1."}],
    )

    assert text == "The question is about a serial number."
    assert len(tool_trace) == 1
    assert tool_trace[0]["tool"] == "get_eval_run"
    assert tool_trace[0]["args"] == {"cycle_id": 62, "query_id": 1}
    assert input_tokens == 200  # summed across both provider.send() calls
    assert output_tokens == 100


@pytest.mark.asyncio
async def test_run_turn_reports_tool_error_without_crashing():
    provider = ScriptedProvider([
        _response(
            [_tool_use_block("t1", "get_eval_run", {"cycle_id": 999, "query_id": 1})],
            "tool_use",
        ),
        _response([_text_block("That cycle doesn't exist.")], "end_turn"),
    ])

    class FailingDatabase:
        async def get_eval_run_detail(self, cycle_id, query_id):
            return None

    text, tool_trace, input_tokens, output_tokens = await run_turn(
        provider, FailingDatabase(), dim=768, log_file="unused.log",
        messages=[{"role": "user", "content": "Investigate cycle 999, question 1."}],
    )

    assert text == "That cycle doesn't exist."
    assert "No eval_cycles row" in tool_trace[0]["result_summary"]
    assert input_tokens == 200
    assert output_tokens == 100


@pytest.mark.asyncio
async def test_run_turn_stops_at_iteration_cap():
    tool_use_response = _response(
        [_tool_use_block("t1", "get_eval_run", {"cycle_id": 62, "query_id": 1})],
        "tool_use",
    )
    provider = ScriptedProvider([tool_use_response] * 8)

    text, tool_trace, input_tokens, output_tokens = await run_turn(
        provider, FakeDatabase(), dim=768, log_file="unused.log",
        messages=[{"role": "user", "content": "Investigate cycle 62, question 1."}],
    )

    assert "tool-call limit" in text
    assert len(tool_trace) == 8
    assert input_tokens == 800  # 8 calls x 100
    assert output_tokens == 400  # 8 calls x 50


def test_to_anthropic_messages_maps_role_and_content():
    history = [
        {"role": "user", "content": "hi", "tool_trace": None},
        {"role": "assistant", "content": "hello", "tool_trace": [{"tool": "get_eval_run"}]},
    ]

    result = to_anthropic_messages(history)

    assert result == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
