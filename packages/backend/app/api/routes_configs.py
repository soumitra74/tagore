from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.db import ExperimentConfig
from app.providers.embedding_catalog import get_embedding_dim

router = APIRouter()

KNOWN_EMBEDDING_PROVIDERS = {"openai", "ollama", "local"}

# The design note's full target set (Section 6) -- all five now implemented.
ALL_CHUNKING_STRATEGIES = {
    "sliding_window",
    "fixed_token",
    "recursive_semantic",
    "sentence_window",
    "document_structure_aware",
}
IMPLEMENTED_CHUNKING_STRATEGIES = ALL_CHUNKING_STRATEGIES


class CreateConfigRequest(BaseModel):
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_provider: str
    embedding_model: str


def validate_config_fields(body: CreateConfigRequest) -> None:
    if body.embedding_provider not in KNOWN_EMBEDDING_PROVIDERS:
        raise HTTPException(
            status_code=400, detail=f"Unknown embedding_provider: {body.embedding_provider!r}"
        )
    if body.chunking_strategy not in ALL_CHUNKING_STRATEGIES:
        raise HTTPException(
            status_code=400, detail=f"Unknown chunking_strategy: {body.chunking_strategy!r}"
        )
    if body.chunking_strategy not in IMPLEMENTED_CHUNKING_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"chunking_strategy {body.chunking_strategy!r} is not yet implemented",
        )


def resolve_embedding_dim(provider: str, model: str) -> int:
    """embedding_dim is a property of the model, not an independent client
    choice -- derive it server-side rather than trusting a client-supplied
    value that could mismatch and break pgvector inserts downstream."""
    dim = get_embedding_dim(provider, model)
    if dim is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown embedding_model {model!r} for provider {provider!r}",
        )
    return dim


def config_to_dict(config: ExperimentConfig) -> dict:
    return {
        "id": config.id,
        "chunking_strategy": config.chunking_strategy,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
    }


@router.post("/api/configs")
async def create_config(request: Request, body: CreateConfigRequest):
    validate_config_fields(body)
    embedding_dim = resolve_embedding_dim(body.embedding_provider, body.embedding_model)
    config = await request.app.state.db.create_experiment_config(
        chunking_strategy=body.chunking_strategy,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
        embedding_provider=body.embedding_provider,
        embedding_model=body.embedding_model,
        embedding_dim=embedding_dim,
    )
    return config_to_dict(config)


@router.get("/api/configs")
async def list_configs(request: Request):
    configs = await request.app.state.db.list_experiment_configs()
    return [config_to_dict(c) for c in configs]
