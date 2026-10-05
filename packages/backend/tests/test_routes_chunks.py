from fastapi.testclient import TestClient

from app.db import ExperimentConfig
from app.main import app

CONFIG = ExperimentConfig(
    id=1,
    chunking_strategy="sliding_window",
    chunk_size=1000,
    chunk_overlap=200,
    embedding_provider="local",
    embedding_model="BAAI/bge-base-en-v1.5",
    embedding_dim=768,
)

DOC_ID = "11111111-1111-1111-1111-111111111111"
CHUNK_ID = "22222222-2222-2222-2222-222222222222"


OTHER_CHUNK_ID = "33333333-3333-3333-3333-333333333333"
MISSING_CHUNK_ID = "44444444-4444-4444-4444-444444444444"


class FakeDatabase:
    def __init__(self):
        self.list_chunk_documents_calls: list[tuple] = []
        self.list_chunks_calls: list[tuple] = []
        self.resolve_chunks_calls: list[tuple] = []

    async def get_experiment_config(self, config_id: int):
        return CONFIG if config_id == CONFIG.id else None

    async def list_chunk_documents(self, config_id: int, dim: int):
        self.list_chunk_documents_calls.append((config_id, dim))
        return [{"document_id": DOC_ID, "filename": "doc.pdf"}]

    async def list_chunks(self, config_id, dim, document_id, search, page, page_size):
        self.list_chunks_calls.append((config_id, dim, document_id, search, page, page_size))
        chunks = [
            {
                "id": CHUNK_ID, "page_number": 3, "chunk_text": "chunk text here",
                "context_text": None, "filename": "doc.pdf",
            }
        ]
        return chunks, 1

    async def resolve_chunks(self, chunk_ids, dim):
        self.resolve_chunks_calls.append((chunk_ids, dim))
        known = {
            CHUNK_ID: {
                "id": CHUNK_ID, "chunk_text": "chunk one", "context_text": None,
                "page_number": 3, "filename": "doc.pdf",
            },
            OTHER_CHUNK_ID: {
                "id": OTHER_CHUNK_ID, "chunk_text": "chunk two", "context_text": "padded context",
                "page_number": 5, "filename": "doc.pdf",
            },
        }
        return {cid: known[cid] for cid in chunk_ids if cid in known}


def test_list_chunk_documents_returns_documents_for_known_config():
    db = FakeDatabase()
    app.state.db = db
    client = TestClient(app)

    response = client.get("/api/chunks/documents", params={"config_id": 1})

    assert response.status_code == 200
    assert response.json() == [{"document_id": DOC_ID, "filename": "doc.pdf"}]
    assert db.list_chunk_documents_calls == [(1, 768)]


def test_list_chunk_documents_404s_for_unknown_config():
    app.state.db = FakeDatabase()
    client = TestClient(app)

    response = client.get("/api/chunks/documents", params={"config_id": 999})

    assert response.status_code == 400
    assert "999" in response.json()["detail"]


def test_list_chunks_returns_paginated_shape():
    db = FakeDatabase()
    app.state.db = db
    client = TestClient(app)

    response = client.get(
        "/api/chunks",
        params={"config_id": 1, "document_id": DOC_ID, "search": "foo", "page": 2, "page_size": 10},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["page"] == 2
    assert body["page_size"] == 10
    [chunk] = body["chunks"]
    assert chunk["id"] == CHUNK_ID
    assert chunk["page_number"] == 3
    assert chunk["chunk_text"] == "chunk text here"
    assert chunk["context_text"] is None
    assert chunk["filename"] == "doc.pdf"

    assert db.list_chunks_calls == [(1, 768, DOC_ID, "foo", 2, 10)]


def test_list_chunks_defaults_page_and_page_size_and_optional_filters():
    db = FakeDatabase()
    app.state.db = db
    client = TestClient(app)

    response = client.get("/api/chunks", params={"config_id": 1})

    assert response.status_code == 200
    assert db.list_chunks_calls == [(1, 768, None, None, 1, 25)]


def test_list_chunks_400s_for_unknown_config():
    app.state.db = FakeDatabase()
    client = TestClient(app)

    response = client.get("/api/chunks", params={"config_id": 999})

    assert response.status_code == 400


def test_list_chunks_rejects_invalid_page():
    app.state.db = FakeDatabase()
    client = TestClient(app)

    response = client.get("/api/chunks", params={"config_id": 1, "page": 0})

    assert response.status_code == 422


def test_lookup_chunks_resolves_comma_separated_ids_preserving_order():
    db = FakeDatabase()
    app.state.db = db
    client = TestClient(app)

    response = client.get(
        "/api/chunks/lookup",
        params={"config_id": 1, "ids": f"{OTHER_CHUNK_ID}, {CHUNK_ID}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["not_found"] == []
    [first, second] = body["chunks"]
    assert first["id"] == OTHER_CHUNK_ID
    assert first["chunk_text"] == "chunk two"
    assert first["context_text"] == "padded context"
    assert second["id"] == CHUNK_ID
    assert second["chunk_text"] == "chunk one"

    assert db.resolve_chunks_calls == [([OTHER_CHUNK_ID, CHUNK_ID], 768)]


def test_lookup_chunks_reports_ids_not_found():
    app.state.db = FakeDatabase()
    client = TestClient(app)

    response = client.get(
        "/api/chunks/lookup", params={"config_id": 1, "ids": f"{CHUNK_ID},{MISSING_CHUNK_ID}"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["not_found"] == [MISSING_CHUNK_ID]
    [chunk] = body["chunks"]
    assert chunk["id"] == CHUNK_ID


def test_lookup_chunks_ignores_blank_entries_and_whitespace():
    db = FakeDatabase()
    app.state.db = db
    client = TestClient(app)

    response = client.get(
        "/api/chunks/lookup", params={"config_id": 1, "ids": f" {CHUNK_ID} ,, "}
    )

    assert response.status_code == 200
    assert db.resolve_chunks_calls == [([CHUNK_ID], 768)]


def test_lookup_chunks_400s_for_unknown_config():
    app.state.db = FakeDatabase()
    client = TestClient(app)

    response = client.get("/api/chunks/lookup", params={"config_id": 999, "ids": CHUNK_ID})

    assert response.status_code == 400
