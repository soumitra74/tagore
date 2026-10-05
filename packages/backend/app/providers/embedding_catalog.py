from app.providers.local_embedding_provider import LOCAL_EMBEDDING_MODELS

# Mirrors packages/frontend/src/api/client.ts's EMBEDDING_MODEL_CATALOG. The
# frontend no longer lets the user pick embedding_dim directly (it's a
# property of the model, not an independent choice) -- this is the single
# source of truth the backend uses to derive/validate it from provider+model.
# "local" reuses LOCAL_EMBEDDING_MODELS (that provider's own source of truth)
# rather than duplicating its dims here.
_OPENAI_EMBEDDING_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_OLLAMA_EMBEDDING_DIMS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}

_PROVIDER_CATALOGS: dict[str, dict[str, int]] = {
    "openai": _OPENAI_EMBEDDING_DIMS,
    "ollama": _OLLAMA_EMBEDDING_DIMS,
}


def get_embedding_dim(provider: str, model: str) -> int | None:
    """The dimension the given provider+model combination produces, or None
    if the combination isn't recognized."""
    if provider == "local":
        spec = LOCAL_EMBEDDING_MODELS.get(model)
        return spec.dim if spec else None
    return _PROVIDER_CATALOGS.get(provider, {}).get(model)
