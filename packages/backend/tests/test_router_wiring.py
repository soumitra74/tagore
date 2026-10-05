import pytest

from app.config import Settings
from app.providers.base import LLMResponse
from app.routing.adapters import RoutedChatProvider, RoutedCompletionProvider
from app.routing.from_settings import create_router, judge_identity
from app.routing.router import DynamicRouter, StaticTableStrategy, TaskType, Tier
from app.services.judge import Judge, create_judge

SMALL = Tier("small", "m-7b", "http://gw/v1")
LARGE = Tier("large", "m-70b", "http://gw/v1")
TIERS = [
    {"name": "small", "model": "m-7b", "base_url": "http://gw/v1"},
    {"name": "large", "model": "m-70b", "base_url": "http://gw/v1"},
]


class RecordingClient:
    def __init__(self):
        self.calls = []

    async def complete(self, tier, messages):
        self.calls.append((tier.name, messages))
        return LLMResponse(text='{"score": 0.8, "reasoning": "fine"}', input_tokens=7, output_tokens=3)


def _router(client, table):
    return DynamicRouter([SMALL, LARGE], StaticTableStrategy(table, "large"), client, "large")


@pytest.mark.asyncio
async def test_routed_chat_provider_uses_synthesis_tier_and_builds_prompt():
    client = RecordingClient()
    provider = RoutedChatProvider(_router(client, {TaskType.SYNTHESIS: "large"}))
    out = await provider.generate("why?", ["[Source 1 -- a.pdf, page 1]\nbecause"])
    [(tier, messages)] = client.calls
    assert tier == "large"
    assert "Question: why?" in messages[0]["content"] and "because" in messages[0]["content"]
    assert (out.input_tokens, out.output_tokens) == (7, 3)


@pytest.mark.asyncio
async def test_routed_completion_provider_uses_judge_tier():
    client = RecordingClient()
    provider = RoutedCompletionProvider(_router(client, {TaskType.JUDGE: "small"}))
    out = await provider.complete("score this")
    assert client.calls == [("small", [{"role": "user", "content": "score this"}])]
    assert out.text.startswith("{")


@pytest.mark.asyncio
async def test_judge_created_with_router_scores_through_it():
    client = RecordingClient()
    judge = create_judge(Settings(_env_file=None), router=_router(client, {TaskType.JUDGE: "small"}))
    assert isinstance(judge, Judge)
    result = await judge.score_correctness("q", "ref", "ans")
    assert result.score == 0.8 and client.calls[0][0] == "small"


def test_create_router_is_none_when_no_tiers():
    assert create_router(Settings(_env_file=None)) is None


def test_create_router_builds_when_tiers_configured():
    s = Settings(_env_file=None, router_tiers=TIERS, router_fallback_tier="large")
    assert isinstance(create_router(s), DynamicRouter)


def test_judge_identity_without_router_is_the_configured_judge():
    s = Settings(_env_file=None, eval_judge_provider="ollama", eval_judge_model="mistral:latest")
    assert judge_identity(s) == ("ollama", "mistral:latest")


def test_judge_identity_with_router_names_the_routed_tier():
    s = Settings(_env_file=None, router_tiers=TIERS, router_fallback_tier="large",
                 router_task_tiers={"judge": "small"})
    assert judge_identity(s) == ("router", "small:m-7b")


def test_judge_identity_with_router_falls_back_when_judge_unmapped():
    s = Settings(_env_file=None, router_tiers=TIERS, router_fallback_tier="large")
    assert judge_identity(s) == ("router", "large:m-70b")
