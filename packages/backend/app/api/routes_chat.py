from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.providers.factory import get_embedding_provider_for_config
from app.services.chat import answer_question

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1, le=20)
    config_id: int | None = None


@router.post("/api/chat")
async def chat(request: Request, body: ChatRequest):
    settings = request.app.state.settings
    db = request.app.state.db
    top_k = body.top_k if body.top_k is not None else settings.retrieval_top_k

    resolved_config_id = (
        body.config_id if body.config_id is not None else await db.get_default_config_id()
    )
    config = await db.get_experiment_config(resolved_config_id)
    if config is None:
        raise HTTPException(status_code=400, detail=f"Unknown config_id: {resolved_config_id}")

    cache = request.app.state.embedding_provider_cache
    if config.id not in cache:
        cache[config.id] = get_embedding_provider_for_config(settings, config)

    try:
        result = await answer_question(
            body.question,
            db,
            cache[config.id],
            request.app.state.chat_provider,
            top_k,
            config,
        )
    except Exception as exc:
        # Covers a missing/invalid provider API key, or an unreachable
        # Ollama server when CHAT_PROVIDER/EMBEDDING_PROVIDER=ollama.
        raise HTTPException(status_code=502, detail=f"Provider request failed: {exc}") from exc

    return {
        "answer": result.answer,
        "sources": [
            {"document": s.document, "page": s.page, "snippet": s.snippet} for s in result.sources
        ],
    }
