from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

router = APIRouter()

DEFAULT_PAGE_SIZE = 25
BACKFILL_PAGE_SIZE = 200


class BackfillRequest(BaseModel):
    config_id: int


async def _iter_all_chunks(db, config_id: int, dim: int, document_id: str):
    page = 1
    while True:
        chunks, total = await db.list_chunks(
            config_id, dim, document_id, None, page, BACKFILL_PAGE_SIZE
        )
        for chunk in chunks:
            yield chunk
        if page * BACKFILL_PAGE_SIZE >= total:
            break
        page += 1


@router.post("/api/graph/backfill")
async def backfill_graph(request: Request, body: BackfillRequest):
    db = request.app.state.db
    ner = request.app.state.ner

    config = await db.get_experiment_config(body.config_id)
    if config is None:
        raise HTTPException(status_code=400, detail=f"Unknown config_id: {body.config_id}")

    documents = await db.list_chunk_documents(body.config_id, config.embedding_dim)
    chunks_processed = 0
    entities_touched: set[str] = set()
    mentions_created = 0
    cooccurrences_created = 0

    for doc in documents:
        document_id = str(doc["document_id"])
        async for chunk in _iter_all_chunks(db, body.config_id, config.embedding_dim, document_id):
            chunks_processed += 1
            extracted = await ner.extract(chunk["chunk_text"])
            for entity in extracted:
                entity_id = await db.insert_entity(entity.entity_type, entity.normalized_text)
                entities_touched.add(entity_id)
                await db.insert_mention(
                    body.config_id, str(chunk["id"]), entity_id,
                    entity.char_start, entity.char_end,
                )
                mentions_created += 1
        cooccurrences_created += await db.derive_cooccurrence_edges(
            body.config_id, config.embedding_dim, document_id
        )

    return {
        "chunks_processed": chunks_processed,
        "entities_touched": len(entities_touched),
        "mentions_created": mentions_created,
        "cooccurrences_created": cooccurrences_created,
    }


@router.get("/api/graph/entities")
async def list_graph_entities(
    request: Request,
    config_id: int,
    document_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
):
    db = request.app.state.db
    config = await db.get_experiment_config(config_id)
    if config is None:
        raise HTTPException(status_code=400, detail=f"Unknown config_id: {config_id}")
    entities, total = await db.list_graph_entities(
        config_id, config.embedding_dim, document_id, page, page_size
    )
    return {
        "entities": [
            {
                "entity_type": e["entity_type"],
                "entity_text": e["entity_text"],
                "chunk_id": str(e["chunk_id"]),
                "page_number": e["page_number"],
                "filename": e["filename"],
            }
            for e in entities
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/api/graph/cooccurrences")
async def list_cooccurrences(
    request: Request,
    document_id: str,
    entity_type: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
):
    db = request.app.state.db
    pairs, total = await db.list_cooccurrences(document_id, entity_type, page, page_size)
    return {
        "cooccurrences": [dict(p) for p in pairs],
        "total": total,
        "page": page,
        "page_size": page_size,
    }
