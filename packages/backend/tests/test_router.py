import logging

import pytest

from app.providers.base import LLMResponse
from app.routing.router import (
    DynamicRouter,
    EscalationPolicy,
    FeatureRuleStrategy,
    RequestFeatures,
    Rule,
    RoutingRequest,
    StaticTableStrategy,
    TaskType,
    Tier,
)

SMALL = Tier("small", "m-7b", "http://gw/v1", cost_per_1k_in=1.0, cost_per_1k_out=2.0)
LARGE = Tier("large", "m-70b", "http://gw/v1", cost_per_1k_in=10.0, cost_per_1k_out=20.0)


class FakeClient:
    def __init__(self, replies: dict[str, list]):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls: list[str] = []

    async def complete(self, tier, messages):
        self.calls.append(tier.name)
        reply = self.replies[tier.name].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _req(task=TaskType.EXTRACTION, **features):
    return RoutingRequest(task, [{"role": "user", "content": "hi"}], RequestFeatures(**features))


def _resp(text="ok", i=1000, o=500):
    return LLMResponse(text=text, input_tokens=i, output_tokens=o)


def _router(client, strategy=None, escalation=None, sink=None):
    strategy = strategy or StaticTableStrategy(
        {TaskType.EXTRACTION: "small", TaskType.SYNTHESIS: "large"}, default="large"
    )
    return DynamicRouter([SMALL, LARGE], strategy, client, "large", escalation, sink)


@pytest.mark.asyncio
async def test_static_table_routes_by_task_and_records_cost():
    client = FakeClient({"small": [_resp()]})
    out = await _router(client).complete(_req(TaskType.EXTRACTION))
    assert client.calls == ["small"]
    assert out.tier == "small" and not out.escalated
    [rec] = out.records
    assert rec.reason == "table: extraction"
    assert rec.cost == pytest.approx(1.0 + 1.0)  # 1k in * 1.0 + 0.5k out * 2.0


@pytest.mark.asyncio
async def test_unmapped_task_uses_default_tier():
    client = FakeClient({"large": [_resp()]})
    out = await _router(client).complete(_req(TaskType.CLASSIFICATION))
    assert out.tier == "large"
    assert "no table entry" in out.records[0].reason


@pytest.mark.asyncio
async def test_feature_rule_overrides_base_strategy():
    base = StaticTableStrategy({TaskType.EXTRACTION: "small"}, default="large")
    rules = [Rule("many docs", lambda r: r.features.n_docs > 3, "large")]
    client = FakeClient({"large": [_resp()]})
    out = await _router(client, FeatureRuleStrategy(rules, base)).complete(_req(n_docs=5))
    assert out.tier == "large"
    assert out.records[0].reason == "rule: many docs"


@pytest.mark.asyncio
async def test_feature_rule_falls_through_to_base_when_no_match():
    base = StaticTableStrategy({TaskType.EXTRACTION: "small"}, default="large")
    rules = [Rule("many docs", lambda r: r.features.n_docs > 3, "large")]
    client = FakeClient({"small": [_resp()]})
    out = await _router(client, FeatureRuleStrategy(rules, base)).complete(_req(n_docs=1))
    assert out.tier == "small"


@pytest.mark.asyncio
async def test_escalates_to_fallback_when_check_fails():
    check = EscalationPolicy(lambda r, resp: "no citation" if "[Source" not in resp.text else None)
    client = FakeClient({"small": [_resp("uncited")], "large": [_resp("cited [Source 1]")]})
    out = await _router(client, escalation=check).complete(_req())
    assert client.calls == ["small", "large"]
    assert out.escalated and out.tier == "large"
    assert out.text == "cited [Source 1]"
    assert out.records[1].escalated and out.records[1].reason == "escalated: no citation"


@pytest.mark.asyncio
async def test_no_escalation_when_check_passes():
    check = EscalationPolicy(lambda r, resp: None)
    client = FakeClient({"small": [_resp()]})
    out = await _router(client, escalation=check).complete(_req())
    assert client.calls == ["small"] and not out.escalated


@pytest.mark.asyncio
async def test_fallback_tier_is_never_escalated_again():
    check = EscalationPolicy(lambda r, resp: "always bad")
    client = FakeClient({"large": [_resp()]})
    out = await _router(client, escalation=check).complete(_req(TaskType.SYNTHESIS))
    assert client.calls == ["large"] and not out.escalated


@pytest.mark.asyncio
async def test_tier_error_falls_back_to_fallback_tier():
    client = FakeClient({"small": [RuntimeError("down")], "large": [_resp()]})
    out = await _router(client).complete(_req())
    assert out.tier == "large"
    [rec] = out.records
    assert rec.escalated and rec.reason == "error on small: RuntimeError"


@pytest.mark.asyncio
async def test_fallback_tier_error_propagates():
    client = FakeClient({"large": [RuntimeError("down")]})
    with pytest.raises(RuntimeError):
        await _router(client).complete(_req(TaskType.SYNTHESIS))


@pytest.mark.asyncio
async def test_strategy_naming_unknown_tier_uses_fallback():
    strategy = StaticTableStrategy({TaskType.EXTRACTION: "ghost"}, default="large")
    client = FakeClient({"large": [_resp()]})
    out = await _router(client, strategy).complete(_req())
    assert out.tier == "large" and "unknown tier" in out.records[0].reason


@pytest.mark.asyncio
async def test_sink_receives_every_record():
    seen = []

    async def sink(rec):
        seen.append(rec)

    check = EscalationPolicy(lambda r, resp: "bad")
    client = FakeClient({"small": [_resp()], "large": [_resp()]})
    await _router(client, escalation=check, sink=sink).complete(_req())
    assert [r.tier for r in seen] == ["small", "large"]


def test_bad_fallback_tier_rejected():
    with pytest.raises(ValueError):
        DynamicRouter([SMALL], StaticTableStrategy({}, "small"), FakeClient({}), "nope")


@pytest.mark.asyncio
async def test_routing_decision_logged_at_info(caplog):
    client = FakeClient({"small": [_resp()]})
    with caplog.at_level(logging.INFO, logger="app.routing.router"):
        await _router(client).complete(_req(TaskType.EXTRACTION))
    [rec] = [r for r in caplog.records if r.name == "app.routing.router"]
    assert rec.levelno == logging.INFO
    msg = rec.getMessage()
    assert "task=extraction" in msg and "tier=small" in msg and "model=m-7b" in msg
    assert "reason='table: extraction'" in msg and "escalated=False" in msg


@pytest.mark.asyncio
async def test_escalation_logs_each_attempt(caplog):
    check = EscalationPolicy(lambda r, resp: "bad")
    client = FakeClient({"small": [_resp()], "large": [_resp()]})
    with caplog.at_level(logging.INFO, logger="app.routing.router"):
        await _router(client, escalation=check).complete(_req())
    msgs = [r.getMessage() for r in caplog.records if r.name == "app.routing.router"]
    assert len(msgs) == 2
    assert "tier=large" in msgs[1] and "escalated=True" in msgs[1]
