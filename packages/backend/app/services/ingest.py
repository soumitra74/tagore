from dataclasses import dataclass

from app.db import ExperimentConfig
from app.services.chunking import chunk_page, extract_json_pages, extract_pages


@dataclass
class IngestResult:
    document_id: str
    chunk_count: int
    page_count: int


def _extract_document_pages(filename: str, raw: bytes) -> list[tuple[int, str]]:
    if filename.lower().endswith(".json"):
        return extract_json_pages(raw)
    return extract_pages(raw)


async def ingest_pdf(
    filename: str, pdf_bytes: bytes, db, embedding_provider, config: ExperimentConfig
) -> IngestResult:
    # Extracted unconditionally, even on a cache hit below, since page_count
    # is a property of the source file, not of this config's chunking -- a
    # re-ingest into a different config still needs the real count, not a
    # count derived from chunks (which could undercount a page that produced
    # zero chunks). JSON collections are split on ---PAGE BREAK--- into the
    # same page list a PDF produces; everything after this point is shared.
    pages = _extract_document_pages(filename, pdf_bytes)
    page_count = len(pages)

    existing_document_id = await db.get_document_by_filename(filename)
    if existing_document_id is not None:
        existing_chunk_count = await db.count_chunks_for_document(
            config.id, existing_document_id, config.embedding_dim
        )
        if existing_chunk_count > 0:
            return IngestResult(
                document_id=existing_document_id, chunk_count=existing_chunk_count,
                page_count=page_count,
            )

    chunks = []
    for page_number, text in pages:
        chunks.extend(
            chunk_page(config.chunking_strategy, page_number, text, config.chunk_size, config.chunk_overlap)
        )

    document_id = existing_document_id or await db.insert_document(filename)

    if not chunks:
        return IngestResult(document_id=document_id, chunk_count=0, page_count=page_count)

    embeddings = await embedding_provider.embed_documents([c.text for c in chunks])
    await db.insert_chunks(document_id, config.id, config.embedding_dim, chunks, embeddings)
    return IngestResult(document_id=document_id, chunk_count=len(chunks), page_count=page_count)
