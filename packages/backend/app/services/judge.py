import json
from dataclasses import dataclass
from typing import Protocol

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import Settings
from app.providers.base import LLMResponse


class CompletionProvider(Protocol):
    async def complete(self, prompt: str) -> LLMResponse: ...


class AnthropicCompletionProvider:
    def __init__(self, settings: Settings):
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=300)
        self._model = settings.eval_judge_model
        self._temperature = settings.eval_judge_temperature

    async def complete(self, prompt: str) -> LLMResponse:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class OpenAICompletionProvider:
    def __init__(self, settings: Settings):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=300)
        self._model = settings.eval_judge_model
        self._temperature = settings.eval_judge_temperature

    async def complete(self, prompt: str) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )


class OllamaCompletionProvider:
    def __init__(self, settings: Settings):
        self._base_url = settings.ollama_base_url
        self._model = settings.eval_judge_model
        self._temperature = settings.eval_judge_temperature
        self._num_ctx = settings.ollama_num_ctx

    async def complete(self, prompt: str) -> LLMResponse:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=300) as client:
            response = await client.post(
                "/api/generate",
                json={
                    "model": self._model, "prompt": prompt, "stream": False,
                    "options": {"temperature": self._temperature, "num_ctx": self._num_ctx},
                },
            )
            response.raise_for_status()
            data = response.json()
            return LLMResponse(
                text=data["response"],
                input_tokens=data.get("prompt_eval_count"),
                output_tokens=data.get("eval_count"),
            )


def get_completion_provider(settings: Settings) -> CompletionProvider:
    if settings.eval_judge_provider == "anthropic":
        return AnthropicCompletionProvider(settings)
    if settings.eval_judge_provider == "openai":
        return OpenAICompletionProvider(settings)
    if settings.eval_judge_provider == "ollama":
        return OllamaCompletionProvider(settings)
    raise ValueError(f"Unknown eval_judge_provider: {settings.eval_judge_provider!r}")


@dataclass
class JudgeResult:
    score: float
    reasoning: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class CombinedJudgeResult:
    """Result of score_reference_based -- one LLM call producing all three
    reference/no-context scores at once (see docs/pending-tasks.md)."""
    correctness: float
    correctness_reasoning: str
    answer_relevance: float
    answer_relevance_reasoning: str
    citation_completeness: float
    citation_completeness_reasoning: str
    input_tokens: int | None = None
    output_tokens: int | None = None


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


class Judge:
    def __init__(self, provider: CompletionProvider):
        self._provider = provider

    async def _score(self, prompt: str) -> JudgeResult:
        response = await self._provider.complete(prompt)
        try:
            data = _extract_json(response.text)
            return JudgeResult(
                score=float(data["score"]),
                reasoning=str(data["reasoning"]),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return JudgeResult(
                score=0.0,
                reasoning=f"Failed to parse judge response: {response.text!r}",
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

    async def score_faithfulness(
        self, question: str, context_chunks: list[str], generated_answer: str
    ) -> JudgeResult:
        context = "\n\n".join(context_chunks) if context_chunks else "(no context retrieved)"
        prompt = (
            "You are checking a RAG system's answer for hallucination.\n\n"
            f"Question: {question}\n"
            f"Retrieved context:\n{context}\n\n"
            f"Generated answer: {generated_answer}\n\n"
            "Score how well every claim in the generated answer is supported by "
            "the retrieved context above, from 0.0 (answer is unsupported by or "
            "contradicts the context) to 1.0 (every claim is grounded in the "
            "context). This is about groundedness, not correctness — an answer "
            "can be fully grounded in the context and still be factually wrong "
            "if the context itself is wrong or insufficient; score groundedness "
            "only.\n\n"
            'Respond with ONLY a JSON object: {"score": <float 0.0-1.0>, '
            '"reasoning": "<one sentence>"}'
        )
        return await self._score(prompt)

    async def score_correctness(
        self, question: str, reference_answer: str, generated_answer: str
    ) -> JudgeResult:
        """Separate-call equivalent of score_reference_based's correctness
        dimension -- kept alongside it so eval_judge_combine_calls=False can
        A/B compare against the combined call (see docs/pending-tasks.md)."""
        prompt = (
            "You are grading a RAG system's answer against a reference answer.\n\n"
            f"Question: {question}\n"
            f"Reference answer: {reference_answer}\n"
            f"Generated answer: {generated_answer}\n\n"
            "Score how well the generated answer captures the same facts as the "
            "reference answer, from 0.0 (completely wrong or unrelated) to 1.0 "
            "(fully correct and equivalent). Minor wording differences don't "
            "matter; missing or contradicted facts do.\n\n"
            'Respond with ONLY a JSON object: {"score": <float 0.0-1.0>, '
            '"reasoning": "<one sentence>"}'
        )
        return await self._score(prompt)

    async def score_answer_relevance(self, question: str, generated_answer: str) -> JudgeResult:
        """Separate-call equivalent of score_reference_based's
        answer_relevance dimension (see score_correctness)."""
        prompt = (
            "You are checking whether a RAG system's answer actually addresses "
            "the question asked.\n\n"
            f"Question: {question}\n"
            f"Generated answer: {generated_answer}\n\n"
            "Score how directly the answer addresses the question, from 0.0 "
            "(off-topic, evasive, or a non-answer) to 1.0 (directly and "
            "completely addresses what was asked). This is about relevance to "
            "the question, not factual correctness — a confident, on-topic "
            "answer can score high here even if it turns out to be wrong.\n\n"
            'Respond with ONLY a JSON object: {"score": <float 0.0-1.0>, '
            '"reasoning": "<one sentence>"}'
        )
        return await self._score(prompt)

    async def score_citation_completeness(
        self, question: str, source_labels: list[str], generated_answer: str
    ) -> JudgeResult:
        """Separate-call equivalent of score_reference_based's
        citation_completeness dimension (see score_correctness)."""
        sources = "\n".join(source_labels) if source_labels else "(no sources retrieved)"
        prompt = (
            "You are checking whether a RAG system's answer cites its sources "
            "for every factual claim it makes.\n\n"
            f"Question: {question}\n"
            f"Available sources:\n{sources}\n\n"
            f"Generated answer: {generated_answer}\n\n"
            "Score what fraction of the factual claims in the answer carry an "
            "explicit citation (e.g. [Source N]) pointing to one of the sources "
            "listed above, from 0.0 (no claims cited, or citations point to "
            "sources that don't exist in the list) to 1.0 (every factual claim "
            "is cited to a real source). A sentence with no factual content "
            "(e.g. \"I don't know\") doesn't need a citation and shouldn't "
            "count against the score.\n\n"
            'Respond with ONLY a JSON object: {"score": <float 0.0-1.0>, '
            '"reasoning": "<one sentence>"}'
        )
        return await self._score(prompt)

    async def score_reference_based(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
        source_labels: list[str],
    ) -> CombinedJudgeResult:
        """Correctness, answer_relevance, and citation_completeness in one
        call -- none of the three need the retrieved context, only the
        question/reference/answer/sources (see docs/pending-tasks.md for why
        faithfulness and info_recall stay separate, context-aware calls)."""
        sources = "\n".join(source_labels) if source_labels else "(no sources retrieved)"
        prompt = (
            "You are grading a RAG system's answer on three independent "
            "dimensions. Score each from 0.0 to 1.0 with a one-sentence "
            "reasoning.\n\n"
            f"Question: {question}\n"
            f"Reference answer: {reference_answer}\n"
            f"Available sources:\n{sources}\n"
            f"Generated answer: {generated_answer}\n\n"
            "1. correctness: how well the generated answer captures the same "
            "facts as the reference answer, from 0.0 (completely wrong or "
            "unrelated) to 1.0 (fully correct and equivalent). Minor wording "
            "differences don't matter; missing or contradicted facts do.\n\n"
            "2. answer_relevance: how directly the answer addresses the "
            "question asked, from 0.0 (off-topic, evasive, or a non-answer) "
            "to 1.0 (directly and completely addresses what was asked). This "
            "is about relevance to the question, not factual correctness — a "
            "confident, on-topic answer can score high here even if it turns "
            "out to be wrong.\n\n"
            "3. citation_completeness: what fraction of the factual claims in "
            "the answer carry an explicit citation (e.g. [Source N]) pointing "
            "to one of the sources listed above, from 0.0 (no claims cited, "
            "or citations point to sources that don't exist in the list) to "
            "1.0 (every factual claim is cited to a real source). A sentence "
            "with no factual content (e.g. \"I don't know\") doesn't need a "
            "citation and shouldn't count against the score.\n\n"
            "Respond with ONLY a JSON object of this exact shape: "
            '{"correctness": {"score": <float 0.0-1.0>, "reasoning": '
            '"<one sentence>"}, "answer_relevance": {"score": <float 0.0-1.0>, '
            '"reasoning": "<one sentence>"}, "citation_completeness": '
            '{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}'
        )
        response = await self._provider.complete(prompt)
        try:
            data = _extract_json(response.text)
            return CombinedJudgeResult(
                correctness=float(data["correctness"]["score"]),
                correctness_reasoning=str(data["correctness"]["reasoning"]),
                answer_relevance=float(data["answer_relevance"]["score"]),
                answer_relevance_reasoning=str(data["answer_relevance"]["reasoning"]),
                citation_completeness=float(data["citation_completeness"]["score"]),
                citation_completeness_reasoning=str(data["citation_completeness"]["reasoning"]),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            failure = f"Failed to parse judge response: {response.text!r}"
            return CombinedJudgeResult(
                correctness=0.0, correctness_reasoning=failure,
                answer_relevance=0.0, answer_relevance_reasoning=failure,
                citation_completeness=0.0, citation_completeness_reasoning=failure,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

    async def score_info_recall(
        self, reference_answer: str, context_chunks: list[str]
    ) -> JudgeResult:
        context = "\n\n".join(context_chunks) if context_chunks else "(no context retrieved)"
        prompt = (
            "You are checking whether retrieved context contains enough "
            "information to fully answer a question.\n\n"
            f"Reference answer (the full, correct answer): {reference_answer}\n"
            f"Retrieved context:\n{context}\n\n"
            "Score what fraction of the facts in the reference answer are "
            "actually present somewhere in the retrieved context, from 0.0 "
            "(none of it is present) to 1.0 (everything needed is present). "
            "This is about retrieval coverage, not about how any answer was "
            "phrased.\n\n"
            'Respond with ONLY a JSON object: {"score": <float 0.0-1.0>, '
            '"reasoning": "<one sentence>"}'
        )
        return await self._score(prompt)


def create_judge(settings: Settings, router=None) -> Judge:
    if router is not None:
        from app.routing.adapters import RoutedCompletionProvider

        return Judge(RoutedCompletionProvider(router))
    return Judge(get_completion_provider(settings))
