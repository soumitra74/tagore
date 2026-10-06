import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.api.routes_configs import (
    CreateConfigRequest,
    config_to_dict,
    resolve_embedding_dim,
    validate_config_fields,
)
from app.db import ExperimentConfig
from app.providers.factory import get_embedding_provider_for_config
from app.services.ingest import ingest_pdf

router = APIRouter()


class IngestCorpusFileRequest(BaseModel):
    filename: str
    config_id: int | None = None


# PDFs and the JSON poetry collections. Both go through ingest_pdf, which
# splits JSON on ---PAGE BREAK--- and PDFs on real pages.
_INGEST_SUFFIXES = {".pdf", ".json"}
_LIBRARY_SUFFIXES = {".pdf", ".json"}


def _list_corpus_filenames(
    corpus_dir: str, suffixes: set[str] | None = None
) -> list[str]:
    allowed = _INGEST_SUFFIXES if suffixes is None else suffixes
    path = Path(corpus_dir)
    if not path.is_dir():
        return []
    return sorted(
        p.name
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in allowed
    )


async def _get_embedding_provider(request: Request, config: ExperimentConfig):
    settings = request.app.state.settings
    cache = request.app.state.embedding_provider_cache
    if config.id not in cache:
        cache[config.id] = get_embedding_provider_for_config(settings, config)
    return cache[config.id]


async def _record_run(
    db, config: ExperimentConfig, results: list[dict], input_size_bytes: int, start: float
) -> dict:
    total_time_ms = int((time.monotonic() - start) * 1000)
    total_chunks = sum(r.get("chunk_count", 0) for r in results)
    total_pages = sum(r.get("page_count", 0) for r in results)
    error_count = sum(1 for r in results if "error" in r)
    vector_db_size_bytes = await db.get_config_vector_storage_bytes(
        config.id, config.embedding_dim
    )

    await db.record_ingestion_run(
        config.id, results, total_chunks, total_time_ms, error_count,
        input_size_bytes, vector_db_size_bytes, total_pages,
    )

    return {
        "chunks": total_chunks,
        "total_pages": total_pages,
        "total_time_ms": total_time_ms,
        "error_count": error_count,
        "input_size_bytes": input_size_bytes,
        "vector_db_size_bytes": vector_db_size_bytes,
    }


@router.get("/api/corpus")
async def list_corpus_files(request: Request):
    return _list_corpus_filenames(
        request.app.state.settings.corpus_dir, suffixes=_LIBRARY_SUFFIXES
    )


@router.post("/api/corpus/ingest")
async def ingest_corpus_file(request: Request, body: IngestCorpusFileRequest):
    if request.app.state.ingestion_in_progress:
        raise HTTPException(status_code=409, detail="An ingestion is already in progress")
    request.app.state.ingestion_in_progress = True
    try:
        settings = request.app.state.settings
        available = _list_corpus_filenames(settings.corpus_dir)
        if body.filename not in available:
            raise HTTPException(status_code=404, detail=f"Unknown corpus file: {body.filename!r}")

        db = request.app.state.db
        resolved_config_id = (
            body.config_id if body.config_id is not None else await db.get_default_config_id()
        )
        config = await db.get_experiment_config(resolved_config_id)
        if config is None:
            raise HTTPException(status_code=400, detail=f"Unknown config_id: {resolved_config_id}")

        provider = await _get_embedding_provider(request, config)
        start = time.monotonic()
        contents = (Path(settings.corpus_dir) / body.filename).read_bytes()

        try:
            result = await ingest_pdf(body.filename, contents, db, provider, config)
            results = [{"filename": body.filename, "document_id": result.document_id,
                        "chunk_count": result.chunk_count, "page_count": result.page_count}]
        except Exception as exc:
            results = [{"filename": body.filename, "error": str(exc)}]
            await _record_run(db, config, results, len(contents), start)
            raise HTTPException(status_code=400, detail=f"Ingestion failed: {exc}") from exc

        metrics = await _record_run(db, config, results, len(contents), start)

        return {
            "document_id": result.document_id,
            "chunk_count": result.chunk_count,
            "page_count": result.page_count,
            "config_id": config.id,
            "metrics": metrics,
        }
    finally:
        request.app.state.ingestion_in_progress = False


@router.post("/api/corpus/ingest-config")
async def ingest_config(request: Request, body: CreateConfigRequest):
    """Upsert an experiment config by its field values and (re-)ingest the
    whole corpus into it: reuses a config if one already matches all six
    fields, wiping its existing chunks first so this call always leaves it
    holding a fresh ingest of whatever's currently in corpus/."""
    validate_config_fields(body)
    embedding_dim = resolve_embedding_dim(body.embedding_provider, body.embedding_model)

    if request.app.state.ingestion_in_progress:
        raise HTTPException(status_code=409, detail="An ingestion is already in progress")
    request.app.state.ingestion_in_progress = True
    try:
        db = request.app.state.db
        existing = await db.find_experiment_config(
            chunking_strategy=body.chunking_strategy,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
            embedding_provider=body.embedding_provider,
            embedding_model=body.embedding_model,
            embedding_dim=embedding_dim,
        )
        if existing is not None:
            config = existing
            await db.delete_chunks_for_config(config.id, config.embedding_dim)
        else:
            config = await db.create_experiment_config(
                chunking_strategy=body.chunking_strategy,
                chunk_size=body.chunk_size,
                chunk_overlap=body.chunk_overlap,
                embedding_provider=body.embedding_provider,
                embedding_model=body.embedding_model,
                embedding_dim=embedding_dim,
            )

        provider = await _get_embedding_provider(request, config)
        settings = request.app.state.settings
        corpus_dir = Path(settings.corpus_dir)

        start = time.monotonic()
        input_size_bytes = 0
        results = []
        for filename in _list_corpus_filenames(settings.corpus_dir):
            try:
                contents = (corpus_dir / filename).read_bytes()
                input_size_bytes += len(contents)
                result = await ingest_pdf(filename, contents, db, provider, config)
                results.append(
                    {"filename": filename, "document_id": result.document_id,
                     "chunk_count": result.chunk_count, "page_count": result.page_count}
                )
            except Exception as exc:
                results.append({"filename": filename, "error": str(exc)})

        metrics = await _record_run(db, config, results, input_size_bytes, start)

        return {"config": config_to_dict(config), "results": results, "metrics": metrics}
    finally:
        request.app.state.ingestion_in_progress = False


@router.get("/api/ingestion-runs")
async def list_ingestion_runs(request: Request):
    runs = await request.app.state.db.list_ingestion_runs()
    return [
        {
            "id": run["id"],
            "ran_at": run["ran_at"].isoformat(),
            "results": run["results"],
            "total_chunks": run["total_chunks"],
            "total_pages": run["total_pages"],
            "total_time_ms": run["total_time_ms"],
            "error_count": run["error_count"],
            "input_size_bytes": run["input_size_bytes"],
            "vector_db_size_bytes": run["vector_db_size_bytes"],
            "config": run["config"],
        }
        for run in runs
    ]
