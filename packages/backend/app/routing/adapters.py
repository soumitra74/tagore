from app.providers.base import LLMResponse, build_prompt
from app.routing.router import DynamicRouter, RequestFeatures, RoutingRequest, TaskType


class RoutedChatProvider:
    """ChatProvider backed by DynamicRouter (task: synthesis)."""

    def __init__(self, router: DynamicRouter):
        self._router = router

    async def generate(self, question: str, context_chunks: list[str]) -> LLMResponse:
        request = RoutingRequest(
            task=TaskType.SYNTHESIS,
            messages=[{"role": "user", "content": build_prompt(question, context_chunks)}],
            features=RequestFeatures(n_context_chunks=len(context_chunks)),
        )
        return await _complete(self._router, request)


class RoutedCompletionProvider:
    """Judge CompletionProvider backed by DynamicRouter (task: judge)."""

    def __init__(self, router: DynamicRouter):
        self._router = router

    async def complete(self, prompt: str) -> LLMResponse:
        request = RoutingRequest(
            task=TaskType.JUDGE, messages=[{"role": "user", "content": prompt}]
        )
        return await _complete(self._router, request)


async def _complete(router: DynamicRouter, request: RoutingRequest) -> LLMResponse:
    routed = await router.complete(request)
    return LLMResponse(
        text=routed.text,
        input_tokens=_sum(r.input_tokens for r in routed.records),
        output_tokens=_sum(r.output_tokens for r in routed.records),
    )


def _sum(values) -> int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None
