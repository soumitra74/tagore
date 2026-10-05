import pytest

from judge import Judge


class FakeProvider:
    def __init__(self, response_text: str):
        self._response_text = response_text

    async def complete(self, prompt: str) -> str:
        return self._response_text


@pytest.mark.asyncio
async def test_score_correctness_parses_valid_json():
    judge = Judge(FakeProvider('{"score": 0.8, "reasoning": "Mostly correct."}'))
    result = await judge.score_correctness("q", "ref", "gen")
    assert result.score == 0.8
    assert result.reasoning == "Mostly correct."


@pytest.mark.asyncio
async def test_score_correctness_strips_markdown_fence():
    judge = Judge(FakeProvider('```json\n{"score": 1.0, "reasoning": "Exact match."}\n```'))
    result = await judge.score_correctness("q", "ref", "gen")
    assert result.score == 1.0
    assert result.reasoning == "Exact match."


@pytest.mark.asyncio
async def test_score_correctness_falls_back_on_unparseable_response():
    judge = Judge(FakeProvider("not json at all"))
    result = await judge.score_correctness("q", "ref", "gen")
    assert result.score == 0.0
    assert "Failed to parse" in result.reasoning
