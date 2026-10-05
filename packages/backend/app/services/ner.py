import asyncio
from dataclasses import dataclass

from gliner import GLiNER

MODEL_NAME = "urchade/gliner_base"
ENTITY_LABELS = ["REPORT_NUMBER", "DATA_COUNTER_ID", "REPORT_DATE", "AIRCRAFT_TYPE"]
SCORE_THRESHOLD = 0.5


def normalize_entity_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


@dataclass
class ExtractedEntity:
    entity_type: str
    entity_text: str
    normalized_text: str
    char_start: int
    char_end: int


class NERService:
    def __init__(self):
        self._model: GLiNER | None = None

    def _get_model(self) -> GLiNER:
        # Lazy load, mirroring SentenceTransformersEmbeddingProvider: constructing
        # this service must stay cheap so unit tests don't trigger a model
        # download just by instantiating the class.
        if self._model is None:
            self._model = GLiNER.from_pretrained(MODEL_NAME)
        return self._model

    def _predict(self, text: str) -> list[ExtractedEntity]:
        model = self._get_model()
        raw = model.predict_entities(text, ENTITY_LABELS, threshold=SCORE_THRESHOLD)
        return [
            ExtractedEntity(
                entity_type=r["label"],
                entity_text=r["text"],
                normalized_text=normalize_entity_text(r["text"]),
                char_start=r["start"],
                char_end=r["end"],
            )
            for r in raw
        ]

    async def extract(self, text: str) -> list[ExtractedEntity]:
        return await asyncio.to_thread(self._predict, text)

    async def warm_up(self) -> None:
        """Force model load at a known point (container startup) instead of
        on the first real request -- see docs/pending-tasks.md's first-load
        freeze finding for local models."""
        await self.extract("warm up")
