from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.providers.factory import get_embedding_provider_for_config
from app.services.ingest import ingest_pdf

router = APIRouter()


@router.post("/api/documents")
async def upload_document(
    request: Request, file: UploadFile = File(...), config_id: int | None = Form(None)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    settings = request.app.state.settings
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=400, detail=f"File exceeds {settings.max_upload_mb}MB limit"
        )

    db = request.app.state.db
    resolved_config_id = config_id if config_id is not None else await db.get_default_config_id()
    config = await db.get_experiment_config(resolved_config_id)
    if config is None:
        raise HTTPException(status_code=400, detail=f"Unknown config_id: {resolved_config_id}")

    cache = request.app.state.embedding_provider_cache
    if config.id not in cache:
        cache[config.id] = get_embedding_provider_for_config(settings, config)

    try:
        result = await ingest_pdf(file.filename, contents, db, cache[config.id], config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {exc}") from exc

    return {
        "document_id": result.document_id,
        "chunk_count": result.chunk_count,
        "config_id": config.id,
    }


@router.get("/api/documents")
async def list_documents(request: Request):
    rows = await request.app.state.db.list_documents()
    return [
        {"id": str(r["id"]), "filename": r["filename"], "uploaded_at": r["uploaded_at"].isoformat()}
        for r in rows
    ]
