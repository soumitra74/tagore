from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()

DEFAULT_PAGE_SIZE = 25


async def _resolve_dim(request: Request, config_id: int) -> int:
    config = await request.app.state.db.get_experiment_config(config_id)
    if config is None:
        raise HTTPException(status_code=400, detail=f"Unknown config_id: {config_id}")
    return config.embedding_dim


@router.get("/api/chunks/documents")
async def list_chunk_documents(request: Request, config_id: int):
    dim = await _resolve_dim(request, config_id)
    documents = await request.app.state.db.list_chunk_documents(config_id, dim)
    return [
        {"document_id": str(d["document_id"]), "filename": d["filename"]} for d in documents
    ]


@router.get("/api/chunks/lookup")
async def lookup_chunks(request: Request, config_id: int, ids: str):
    """Resolve a comma-separated list of chunk ids directly, bypassing
    pagination/filters -- for cross-checking a specific id surfaced
    elsewhere (e.g. an eval run's retrieved_chunk_ids, or a citation)
    against what's actually stored."""
    dim = await _resolve_dim(request, config_id)
    requested = [chunk_id.strip() for chunk_id in ids.split(",") if chunk_id.strip()]
    resolved = await request.app.state.db.resolve_chunks(requested, dim)

    chunks = []
    not_found = []
    for chunk_id in requested:
        row = resolved.get(chunk_id)
        if row is None:
            not_found.append(chunk_id)
            continue
        chunks.append(
            {
                "id": str(row["id"]),
                "page_number": row["page_number"],
                "chunk_text": row["chunk_text"],
                "context_text": row["context_text"],
                "filename": row["filename"],
            }
        )
    return {"chunks": chunks, "not_found": not_found}


@router.get("/api/chunks")
async def list_chunks(
    request: Request,
    config_id: int,
    document_id: str | None = None,
    search: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
):
    dim = await _resolve_dim(request, config_id)
    chunks, total = await request.app.state.db.list_chunks(
        config_id, dim, document_id, search, page, page_size
    )
    return {
        "chunks": [
            {
                "id": str(c["id"]),
                "page_number": c["page_number"],
                "chunk_text": c["chunk_text"],
                "context_text": c["context_text"],
                "filename": c["filename"],
            }
            for c in chunks
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }
