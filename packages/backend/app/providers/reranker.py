import asyncio
from typing import Protocol


class RerankerProvider(Protocol):
    async def rerank(self, query: str, passages: list[str]) -> list[float]: ...


# Local cross-encoders (sentence-transformers CrossEncoder, HF cache like the
# local embedders). Value is the max input length in tokens for a
# (query, passage) pair.
RERANKER_MODELS: dict[str, int] = {
    "BAAI/bge-reranker-base": 512,
    "BAAI/bge-reranker-v2-m3": 8192,
}

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANK_CANDIDATES = 20
MAX_RERANK_CANDIDATES = 50


class LocalCrossEncoderReranker:
    def __init__(self, model_name: str):
        if model_name not in RERANKER_MODELS:
            raise ValueError(
                f"Unknown rerank model {model_name!r}; known: {sorted(RERANKER_MODELS)}"
            )
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name, max_length=RERANKER_MODELS[self._model_name]
            )
        return self._model

    def _score(self, query: str, passages: list[str]) -> list[float]:
        scores = self._load().predict([(query, p) for p in passages])
        return [float(s) for s in scores]

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        # Model load and scoring are blocking; keep them off the event loop.
        return await asyncio.to_thread(self._score, query, passages)


def get_reranker(cache: dict, model_name: str) -> RerankerProvider:
    if model_name not in cache:
        cache[model_name] = LocalCrossEncoderReranker(model_name)
    return cache[model_name]
