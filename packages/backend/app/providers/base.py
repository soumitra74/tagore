from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class ChatProvider(Protocol):
    async def generate(self, question: str, context_chunks: list[str]) -> LLMResponse: ...


class EmbeddingProvider(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    return (
        "Answer the question using only the context below. Each context passage "
        "is labeled [Source N -- document, page P]. After every factual claim "
        "you make, cite the source it came from in the exact form [Source N]. "
        "If the context doesn't contain the answer, say so explicitly.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
