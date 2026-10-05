from fastapi.testclient import TestClient

from app.db import ExperimentConfig
from app.main import app

TEST_CONFIG = ExperimentConfig(
    id=15, chunking_strategy="recursive_semantic", chunk_size=512, chunk_overlap=77,
    embedding_provider="local", embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=768,
)


class FakeNER:
    def __init__(self, results_by_text):
        self._results_by_text = results_by_text

    async def extract(self, text):
        return self._results_by_text.get(text, [])


class FakeExtractedEntity:
    def __init__(self, entity_type, normalized_text, char_start=0, char_end=0):
        self.entity_type = entity_type
        self.normalized_text = normalized_text
        self.char_start = char_start
        self.char_end = char_end


class FakeDatabase:
    def __init__(self):
        self.mentions = []
        self.cooccurrence_calls = []
        self.list_cooccurrences_calls = []

    async def get_experiment_config(self, config_id):
        return TEST_CONFIG if config_id == 15 else None

    async def list_chunk_documents(self, config_id, dim):
        return [{"document_id": "doc-1", "filename": "PFR1.pdf"}]

    async def list_chunks(self, config_id, dim, document_id, search, page, page_size):
        if page > 1:
            return [], 1
        return (
            [{"id": "chunk-1", "page_number": 18, "chunk_text": "For Counter 9017...",
              "context_text": None, "filename": "PFR1.pdf"}],
            1,
        )

    async def insert_entity(self, entity_type, normalized_text):
        return f"entity-{entity_type}-{normalized_text}"

    async def insert_mention(self, config_id, chunk_id, entity_id, char_start, char_end):
        self.mentions.append((config_id, chunk_id, entity_id))

    async def derive_cooccurrence_edges(self, config_id, dim, document_id):
        self.cooccurrence_calls.append((config_id, dim, document_id))
        return 2

    async def list_graph_entities(self, config_id, dim, document_id, page, page_size):
        return (
            [{"entity_type": "DATA_COUNTER_ID", "entity_text": "9017", "chunk_id": "chunk-1",
              "page_number": 18, "filename": "PFR1.pdf"}],
            1,
        )

    async def list_cooccurrences(self, document_id, entity_type, page, page_size):
        self.list_cooccurrences_calls.append((document_id, entity_type, page, page_size))
        return (
            [{"entity_type_a": "DATA_COUNTER_ID", "entity_text_a": "9017",
              "entity_type_b": "REPORT_NUMBER", "entity_text_b": "tm-104598"}],
            1,
        )


def test_backfill_extracts_entities_and_derives_cooccurrences():
    db = FakeDatabase()
    app.state.db = db
    app.state.ner = FakeNER({
        "For Counter 9017...": [FakeExtractedEntity("DATA_COUNTER_ID", "9017", 4, 8)],
    })
    client = TestClient(app)

    response = client.post("/api/graph/backfill", json={"config_id": 15})

    assert response.status_code == 200
    body = response.json()
    assert body["chunks_processed"] == 1
    assert body["entities_touched"] == 1
    assert body["mentions_created"] == 1
    assert body["cooccurrences_created"] == 2
    assert db.mentions == [(15, "chunk-1", "entity-DATA_COUNTER_ID-9017")]
    assert db.cooccurrence_calls == [(15, 768, "doc-1")]


def test_backfill_unknown_config_returns_400():
    app.state.db = FakeDatabase()
    app.state.ner = FakeNER({})
    client = TestClient(app)

    response = client.post("/api/graph/backfill", json={"config_id": 999})

    assert response.status_code == 400


def test_list_graph_entities_endpoint():
    app.state.db = FakeDatabase()
    app.state.ner = FakeNER({})
    client = TestClient(app)

    response = client.get(
        "/api/graph/entities", params={"config_id": 15, "document_id": "doc-1"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["entities"][0]["entity_text"] == "9017"


def test_list_cooccurrences_endpoint():
    db = FakeDatabase()
    app.state.db = db
    app.state.ner = FakeNER({})
    client = TestClient(app)

    response = client.get("/api/graph/cooccurrences", params={"document_id": "doc-1"})

    assert response.status_code == 200
    assert db.list_cooccurrences_calls == [("doc-1", None, 1, 25)]


def test_list_cooccurrences_endpoint_filters_by_entity_type():
    db = FakeDatabase()
    app.state.db = db
    app.state.ner = FakeNER({})
    client = TestClient(app)

    response = client.get(
        "/api/graph/cooccurrences",
        params={"document_id": "doc-1", "entity_type": "DATA_COUNTER_ID"},
    )

    assert response.status_code == 200
    assert db.list_cooccurrences_calls == [("doc-1", "DATA_COUNTER_ID", 1, 25)]
    body = response.json()
    assert body["total"] == 1
    assert body["cooccurrences"][0]["entity_text_a"] == "9017"
