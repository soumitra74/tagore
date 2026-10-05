import pytest

from app.providers.base import LLMResponse
from app.services.judge import Judge


class FakeCompletionProvider:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.received_prompt: str | None = None

    async def complete(self, prompt: str) -> LLMResponse:
        self.received_prompt = prompt
        return LLMResponse(text=self.response_text, input_tokens=50, output_tokens=10)


@pytest.mark.asyncio
async def test_score_faithfulness_parses_json_response():
    provider = FakeCompletionProvider('{"score": 0.5, "reasoning": "partially grounded"}')
    judge = Judge(provider)

    result = await judge.score_faithfulness("q?", ["context chunk one"], "generated answer")

    assert result.score == 0.5
    assert result.reasoning == "partially grounded"
    assert "context chunk one" in provider.received_prompt
    assert "generated answer" in provider.received_prompt


@pytest.mark.asyncio
async def test_score_info_recall_parses_json_response():
    provider = FakeCompletionProvider('{"score": 0.25, "reasoning": "context missing most facts"}')
    judge = Judge(provider)

    result = await judge.score_info_recall("reference answer", ["context chunk one"])

    assert result.score == 0.25
    assert result.reasoning == "context missing most facts"
    assert "reference answer" in provider.received_prompt
    assert "context chunk one" in provider.received_prompt


@pytest.mark.asyncio
async def test_score_reference_based_parses_combined_json_response():
    provider = FakeCompletionProvider(
        '{"correctness": {"score": 0.8, "reasoning": "mostly right"}, '
        '"answer_relevance": {"score": 1.0, "reasoning": "directly answers"}, '
        '"citation_completeness": {"score": 0.67, "reasoning": "one claim uncited"}}'
    )
    judge = Judge(provider)

    result = await judge.score_reference_based(
        "q?", "reference", "generated", ["Source 1: doc.pdf, page 3"]
    )

    assert result.correctness == 0.8
    assert result.correctness_reasoning == "mostly right"
    assert result.answer_relevance == 1.0
    assert result.answer_relevance_reasoning == "directly answers"
    assert result.citation_completeness == 0.67
    assert result.citation_completeness_reasoning == "one claim uncited"
    assert result.input_tokens == 50
    assert result.output_tokens == 10
    assert "reference" in provider.received_prompt
    assert "generated" in provider.received_prompt
    assert "Source 1: doc.pdf, page 3" in provider.received_prompt


@pytest.mark.asyncio
async def test_score_reference_based_handles_unparseable_response():
    provider = FakeCompletionProvider("not json at all")
    judge = Judge(provider)

    result = await judge.score_reference_based("q?", "reference", "generated", [])

    assert result.correctness == 0.0
    assert "Failed to parse" in result.correctness_reasoning
    assert result.answer_relevance == 0.0
    assert "Failed to parse" in result.answer_relevance_reasoning
    assert result.citation_completeness == 0.0
    assert "Failed to parse" in result.citation_completeness_reasoning
    assert result.input_tokens == 50
    assert result.output_tokens == 10


@pytest.mark.asyncio
async def test_score_reference_based_handles_partially_malformed_json():
    # Valid JSON but missing a required key -- should fail closed the same
    # way as unparseable text, not raise.
    provider = FakeCompletionProvider('{"correctness": {"score": 0.8, "reasoning": "ok"}}')
    judge = Judge(provider)

    result = await judge.score_reference_based("q?", "reference", "generated", [])

    assert result.correctness == 0.0
    assert result.answer_relevance == 0.0
    assert result.citation_completeness == 0.0


@pytest.mark.asyncio
async def test_score_handles_unparseable_response():
    provider = FakeCompletionProvider("not json at all")
    judge = Judge(provider)

    result = await judge.score_faithfulness("q?", ["ctx"], "answer")

    assert result.score == 0.0
    assert "Failed to parse" in result.reasoning
