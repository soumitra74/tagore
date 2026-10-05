import json
from dataclasses import dataclass

from config import Settings
from llm_client import CompletionProvider, get_completion_provider


@dataclass
class JudgeResult:
    score: float
    reasoning: str


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

    async def score_correctness(
        self, question: str, reference_answer: str, generated_answer: str
    ) -> JudgeResult:
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
        raw = await self._provider.complete(prompt)
        try:
            data = _extract_json(raw)
            return JudgeResult(score=float(data["score"]), reasoning=str(data["reasoning"]))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return JudgeResult(score=0.0, reasoning=f"Failed to parse judge response: {raw!r}")


def create_judge(settings: Settings) -> Judge:
    return Judge(get_completion_provider(settings))
