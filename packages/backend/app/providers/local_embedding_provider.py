import asyncio
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from app.config import Settings


@dataclass
class ModelSpec:
    dim: int
    query_prefix: str
    passage_prefix: str
    trust_remote_code: bool
    # Overrides sentence-transformers' auto-detected max_seq_length when set.
    # Needed for bge-m3: its own config caps out well short of the 8192 tokens
    # it's actually trained for, so the ceiling has to be raised explicitly to
    # get the long-context behavior this model is being added for in the
    # first place -- silently leaving it at the auto-detected default would
    # make this indistinguishable from any other 512-token model.
    max_seq_length: int | None = None


LOCAL_EMBEDDING_MODELS: dict[str, ModelSpec] = {
    "BAAI/bge-base-en-v1.5": ModelSpec(
        dim=768,
        query_prefix="Represent this sentence for searching relevant passages: ",
        passage_prefix="",
        trust_remote_code=False,
    ),
    "intfloat/e5-base-v2": ModelSpec(
        dim=768,
        query_prefix="query: ",
        passage_prefix="passage: ",
        trust_remote_code=False,
    ),
    "nomic-ai/nomic-embed-text-v1.5": ModelSpec(
        dim=768,
        query_prefix="search_query: ",
        passage_prefix="search_document: ",
        trust_remote_code=True,
    ),
    # Long-context option (8192 tokens vs. every other local model's 512-token
    # ceiling): lets chunk_size be swept well beyond what bge-base-en-v1.5
    # would silently truncate. Dense-only here -- BGE-M3 also produces sparse
    # and ColBERT-style multi-vector outputs, but this provider (and the
    # single-dense-vector chunks_dim_N schema) only uses the dense one.
    # Unlike the bge-base v1.5 line, BGE-M3 doesn't need an instruction-style
    # query prefix.
    "BAAI/bge-m3": ModelSpec(
        dim=1024,
        query_prefix="",
        passage_prefix="",
        trust_remote_code=False,
        max_seq_length=8192,
    ),
}


class SentenceTransformersEmbeddingProvider:
    def __init__(self, settings: Settings):
        model_name = settings.local_embedding_model
        if model_name not in LOCAL_EMBEDDING_MODELS:
            raise ValueError(f"Unknown LOCAL_EMBEDDING_MODEL: {model_name!r}")
        self._model_name = model_name
        self._spec = LOCAL_EMBEDDING_MODELS[model_name]
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        # Lazy load: constructing this provider must stay cheap (no network/
        # disk model download) so factory-selection unit tests don't trigger
        # a multi-GB download just by instantiating the class.
        if self._model is None:
            self._model = SentenceTransformer(
                self._model_name, trust_remote_code=self._spec.trust_remote_code
            )
            if self._spec.max_seq_length is not None:
                self._model.max_seq_length = self._spec.max_seq_length
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [embedding.tolist() for embedding in embeddings]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self._spec.passage_prefix + text for text in texts]
        return await asyncio.to_thread(self._encode, prefixed)

    async def embed_query(self, text: str) -> list[float]:
        [embedding] = await asyncio.to_thread(
            self._encode, [self._spec.query_prefix + text]
        )
        return embedding
