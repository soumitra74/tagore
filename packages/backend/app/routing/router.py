import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Protocol

from app.providers.base import LLMResponse

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    SYNTHESIS = "synthesis"
    CRITIQUE = "critique"
    JUDGE = "judge"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"


@dataclass
class RequestFeatures:
    n_context_chunks: int = 0
    n_docs: int = 0
    prompt_tokens: int | None = None


@dataclass
class RoutingRequest:
    task: TaskType
    messages: list[dict]
    features: RequestFeatures = field(default_factory=RequestFeatures)


@dataclass
class Tier:
    name: str
    model: str
    base_url: str
    api_key: str | None = None
    temperature: float = 0.0
    cost_per_1k_in: float = 0.0
    cost_per_1k_out: float = 0.0


@dataclass
class RoutingDecision:
    tier: str
    reason: str


@dataclass
class DecisionRecord:
    task: str
    tier: str
    reason: str
    escalated: bool
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    cost: float


@dataclass
class RoutedResponse:
    text: str
    tier: str
    records: list[DecisionRecord]

    @property
    def escalated(self) -> bool:
        return len(self.records) > 1


class CompletionClient(Protocol):
    async def complete(self, tier: Tier, messages: list[dict]) -> LLMResponse: ...


class RoutingStrategy(Protocol):
    def choose(self, req: RoutingRequest, tiers: dict[str, Tier]) -> RoutingDecision: ...


class StaticTableStrategy:
    def __init__(self, table: dict[TaskType, str], default: str):
        self._table = table
        self._default = default

    def choose(self, req, tiers):
        tier = self._table.get(req.task)
        if tier is None:
            return RoutingDecision(self._default, f"no table entry for {req.task.value}")
        return RoutingDecision(tier, f"table: {req.task.value}")


@dataclass
class Rule:
    name: str
    matches: Callable[[RoutingRequest], bool]
    tier: str


class FeatureRuleStrategy:
    """First matching rule wins; otherwise defers to the base strategy."""

    def __init__(self, rules: list[Rule], base: RoutingStrategy):
        self._rules = rules
        self._base = base

    def choose(self, req, tiers):
        for rule in self._rules:
            if rule.matches(req):
                return RoutingDecision(rule.tier, f"rule: {rule.name}")
        return self._base.choose(req, tiers)


@dataclass
class EscalationPolicy:
    """`check` returns a failure reason to escalate, or None to accept the answer."""

    check: Callable[[RoutingRequest, LLMResponse], str | None]


DecisionSink = Callable[[DecisionRecord], Awaitable[None]]


class DynamicRouter:
    def __init__(
        self,
        tiers: list[Tier],
        strategy: RoutingStrategy,
        client: CompletionClient,
        fallback_tier: str,
        escalation: EscalationPolicy | None = None,
        sink: DecisionSink | None = None,
    ):
        self._tiers = {t.name: t for t in tiers}
        if fallback_tier not in self._tiers:
            raise ValueError(f"fallback_tier {fallback_tier!r} is not a configured tier")
        self._strategy = strategy
        self._client = client
        self._fallback = fallback_tier
        self._escalation = escalation
        self._sink = sink

    async def route(self, req: RoutingRequest) -> RoutingDecision:
        decision = self._strategy.choose(req, self._tiers)
        if decision.tier not in self._tiers:
            return RoutingDecision(
                self._fallback, f"unknown tier {decision.tier!r} ({decision.reason})"
            )
        return decision

    async def complete(self, req: RoutingRequest) -> RoutedResponse:
        decision = await self.route(req)
        records: list[DecisionRecord] = []

        try:
            response = await self._call(req, decision, records, escalated=False)
        except Exception as exc:
            if decision.tier == self._fallback:
                raise
            decision = RoutingDecision(
                self._fallback, f"error on {decision.tier}: {type(exc).__name__}"
            )
            response = await self._call(req, decision, records, escalated=True)
            return RoutedResponse(response.text, decision.tier, records)

        if self._escalation and decision.tier != self._fallback:
            failure = self._escalation.check(req, response)
            if failure:
                decision = RoutingDecision(self._fallback, f"escalated: {failure}")
                response = await self._call(req, decision, records, escalated=True)

        return RoutedResponse(response.text, decision.tier, records)

    async def _call(self, req, decision, records, escalated) -> LLMResponse:
        tier = self._tiers[decision.tier]
        start = time.monotonic()
        response = await self._client.complete(tier, req.messages)
        record = DecisionRecord(
            task=req.task.value,
            tier=tier.name,
            reason=decision.reason,
            escalated=escalated,
            latency_ms=int((time.monotonic() - start) * 1000),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=_cost(tier, response),
        )
        records.append(record)
        logger.info(
            "routed task=%s tier=%s model=%s reason=%r escalated=%s latency_ms=%d "
            "tokens_in=%s tokens_out=%s cost=%.6f",
            record.task, record.tier, tier.model, record.reason, record.escalated,
            record.latency_ms, record.input_tokens, record.output_tokens, record.cost,
        )
        if self._sink:
            await self._sink(record)
        return response


def _cost(tier: Tier, response: LLMResponse) -> float:
    return (
        (response.input_tokens or 0) / 1000 * tier.cost_per_1k_in
        + (response.output_tokens or 0) / 1000 * tier.cost_per_1k_out
    )
