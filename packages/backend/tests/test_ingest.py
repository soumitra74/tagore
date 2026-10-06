import json

import pytest

from app.db import ExperimentConfig
from app.services.chunking import chunk_page
from app.services.ingest import ingest_pdf


class FakeEmbeddingProvider:
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class FakeDatabase:
    def __init__(self, existing_filenames: dict[str, str] | None = None, existing_chunk_counts=None):
        self.documents: list[str] = []
        self.inserted_chunks = None
        self.insert_document_calls = 0
        self._filename_to_id = dict(existing_filenames or {})
        self._chunk_counts = dict(existing_chunk_counts or {})

    async def insert_document(self, filename: str) -> str:
        self.insert_document_calls += 1
        self.documents.append(filename)
        return f"doc-{len(self.documents)}"

    async def insert_chunks(self, document_id, config_id, dim, chunks, embeddings):
        self.inserted_chunks = (document_id, config_id, dim, chunks, embeddings)

    async def get_document_by_filename(self, filename: str) -> str | None:
        return self._filename_to_id.get(filename)

    async def count_chunks_for_document(self, config_id, document_id, dim) -> int:
        return self._chunk_counts.get((config_id, document_id), 0)


TEST_CONFIG = ExperimentConfig(
    id=1,
    chunking_strategy="sliding_window",
    chunk_size=1000,
    chunk_overlap=200,
    embedding_provider="local",
    embedding_model="BAAI/bge-base-en-v1.5",
    embedding_dim=768,
)


@pytest.mark.asyncio
async def test_ingest_pdf_stores_chunks_and_embeddings(monkeypatch):
    def fake_extract_pages(pdf_bytes: bytes):
        return [(1, "some text on page one"), (2, "more text on page two")]

    monkeypatch.setattr("app.services.ingest.extract_pages", fake_extract_pages)

    db = FakeDatabase()
    result = await ingest_pdf(
        "doc.pdf", b"fake-pdf-bytes", db, FakeEmbeddingProvider(), TEST_CONFIG
    )

    assert result.document_id == "doc-1"
    assert result.chunk_count == 2
    assert result.page_count == 2
    document_id, config_id, dim, chunks, embeddings = db.inserted_chunks
    assert document_id == "doc-1"
    assert config_id == 1
    assert dim == 768
    assert [c.page_number for c in chunks] == [1, 2]
    assert len(embeddings) == 2


@pytest.mark.asyncio
async def test_ingest_pdf_with_no_extractable_text_stores_zero_chunks(monkeypatch):
    monkeypatch.setattr("app.services.ingest.extract_pages", lambda pdf_bytes: [(1, "")])

    db = FakeDatabase()
    result = await ingest_pdf(
        "blank.pdf", b"fake-pdf-bytes", db, FakeEmbeddingProvider(), TEST_CONFIG
    )

    assert result.chunk_count == 0
    assert result.page_count == 1
    assert db.inserted_chunks is None


@pytest.mark.asyncio
async def test_ingest_pdf_uses_configs_chunk_size_and_overlap(monkeypatch):
    monkeypatch.setattr(
        "app.services.ingest.extract_pages", lambda pdf_bytes: [(1, "x" * 100)]
    )
    small_config = ExperimentConfig(
        id=2,
        chunking_strategy="sliding_window",
        chunk_size=30,
        chunk_overlap=5,
        embedding_provider="local",
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dim=768,
    )

    db = FakeDatabase()
    await ingest_pdf("doc.pdf", b"fake-pdf-bytes", db, FakeEmbeddingProvider(), small_config)

    _, _, _, chunks, _ = db.inserted_chunks
    # chunk_size=30, overlap=5 over 100 chars produces more, smaller chunks
    # than the default 1000/200 would.
    assert len(chunks) > 1
    assert all(len(c.text) <= 30 for c in chunks)


@pytest.mark.asyncio
async def test_ingest_pdf_respects_configured_chunking_strategy(monkeypatch):
    monkeypatch.setattr(
        "app.services.ingest.extract_pages",
        lambda pdf_bytes: [(1, "the quick brown fox jumps over the lazy dog. " * 40)],
    )
    fixed_token_config = ExperimentConfig(
        id=3,
        chunking_strategy="fixed_token",
        chunk_size=100,
        chunk_overlap=20,
        embedding_provider="local",
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dim=768,
    )

    db = FakeDatabase()
    await ingest_pdf(
        "doc.pdf", b"fake-pdf-bytes", db, FakeEmbeddingProvider(), fixed_token_config
    )

    _, _, _, chunks, _ = db.inserted_chunks
    expected = chunk_page("fixed_token", 1, "the quick brown fox jumps over the lazy dog. " * 40, 100, 20)
    assert chunks == expected


@pytest.mark.asyncio
async def test_ingest_pdf_skips_reingestion_when_already_chunked_for_config(monkeypatch):
    def fake_extract_pages(pdf_bytes: bytes):
        return [(1, "some text on page one")]

    monkeypatch.setattr("app.services.ingest.extract_pages", fake_extract_pages)

    db = FakeDatabase(
        existing_filenames={"doc.pdf": "doc-1"},
        existing_chunk_counts={(1, "doc-1"): 5},
    )
    result = await ingest_pdf(
        "doc.pdf", b"fake-pdf-bytes", db, FakeEmbeddingProvider(), TEST_CONFIG
    )

    assert result.document_id == "doc-1"
    assert result.chunk_count == 5
    assert result.page_count == 1
    assert db.insert_document_calls == 0
    assert db.inserted_chunks is None


@pytest.mark.asyncio
async def test_ingest_pdf_reuses_existing_document_id_for_new_config(monkeypatch):
    def fake_extract_pages(pdf_bytes: bytes):
        return [(1, "some text on page one")]

    monkeypatch.setattr("app.services.ingest.extract_pages", fake_extract_pages)

    other_config = ExperimentConfig(
        id=2,
        chunking_strategy="sliding_window",
        chunk_size=1000,
        chunk_overlap=200,
        embedding_provider="local",
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dim=768,
    )
    db = FakeDatabase(existing_filenames={"doc.pdf": "doc-1"})
    result = await ingest_pdf(
        "doc.pdf", b"fake-pdf-bytes", db, FakeEmbeddingProvider(), other_config
    )

    assert result.document_id == "doc-1"
    assert result.chunk_count == 1
    assert db.insert_document_calls == 0
    document_id, config_id, _, _, _ = db.inserted_chunks
    assert document_id == "doc-1"
    assert config_id == 2


@pytest.mark.asyncio
async def test_ingest_json_collection_chunks_poem_pages():
    raw = json.dumps(
        [
            {
                "poem_number": "ভূমিকা",
                "collection": "আকাশপ্রদীপ",
                "pages": 2,
                "content": "প্রথম পাতা।\n---PAGE BREAK---\nদ্বিতীয় পাতা।",
            }
        ],
        ensure_ascii=False,
    ).encode()

    db = FakeDatabase()
    result = await ingest_pdf(
        "আকাশপ্রদীপ.json", raw, db, FakeEmbeddingProvider(), TEST_CONFIG
    )

    assert result.page_count == 2
    assert result.chunk_count == 2
    chunks = db.inserted_chunks[3]
    assert chunks[0].page_number == 1
    assert chunks[0].text == "ভূমিকা\nপ্রথম পাতা।"
    assert chunks[1].page_number == 2
    assert chunks[1].text == "ভূমিকা\nদ্বিতীয় পাতা।"
