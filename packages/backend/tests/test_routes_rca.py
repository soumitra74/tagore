from fastapi.testclient import TestClient

from app.config import get_settings
from app.db import ExperimentConfig
from app.main import app

TEST_CONFIG = ExperimentConfig(
    id=13, chunking_strategy="sliding_window", chunk_size=1000, chunk_overlap=450,
    embedding_provider="local", embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=768,
)


class FakeProvider:
    pass  # run_turn is monkeypatched per-test; the route never calls provider methods directly.


class FakeDatabase:
    def __init__(self):
        self.sessions = {}
        self.messages = {}
        self._next_session_id = 1

    async def get_eval_cycle_summary(self, cycle_id):
        return {"id": cycle_id, "config_id": 13, "total": 10, "status": "completed"} if cycle_id == 62 else None

    async def get_experiment_config(self, config_id):
        return TEST_CONFIG if config_id == 13 else None

    async def create_rca_session(self, cycle_id, query_id):
        session_id = self._next_session_id
        self._next_session_id += 1
        self.sessions[session_id] = {"id": session_id, "cycle_id": cycle_id, "query_id": query_id}
        self.messages[session_id] = []
        return session_id

    async def get_rca_session(self, session_id):
        return self.sessions.get(session_id)

    async def list_rca_sessions(self):
        return [
            {
                **s,
                "total_input_tokens": sum(m.get("input_tokens") or 0 for m in self.messages[s["id"]]),
                "total_output_tokens": sum(m.get("output_tokens") or 0 for m in self.messages[s["id"]]),
            }
            for s in self.sessions.values()
        ]

    async def soft_delete_rca_session(self, session_id):
        if session_id not in self.sessions:
            return False
        del self.sessions[session_id]
        del self.messages[session_id]
        return True

    async def insert_rca_message(
        self, session_id, role, content, tool_trace, input_tokens=None, output_tokens=None,
    ):
        self.messages[session_id].append({
            "role": role, "content": content, "tool_trace": tool_trace,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        })

    async def list_rca_messages(self, session_id):
        return self.messages[session_id]

    async def list_eval_queries(self):
        return [{"id": 1, "query_text": "serial number?", "query_type": "extractive"}]


def _patch_run_turn(monkeypatch, text="Found the issue.", tool_trace=None, input_tokens=100, output_tokens=50):
    async def fake_run_turn(provider, db, dim, log_file, messages):
        return (
            text, tool_trace or [{"tool": "get_eval_run", "args": {}, "result_summary": "ok"}],
            input_tokens, output_tokens,
        )

    monkeypatch.setattr("app.api.routes_rca.run_turn", fake_run_turn)


def test_create_session_runs_first_turn_and_persists(monkeypatch):
    db = FakeDatabase()
    app.state.db = db
    app.state.rca_provider = FakeProvider()
    app.state.settings = get_settings()
    _patch_run_turn(monkeypatch)
    client = TestClient(app)

    response = client.post("/api/rca/sessions", json={"cycle_id": 62, "query_id": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == 1
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"] == "Found the issue."
    assert len(db.messages[1]) == 2  # synthetic user seed + assistant reply
    assert db.messages[1][0]["role"] == "user"
    assert db.messages[1][1]["role"] == "assistant"


def test_create_session_unknown_cycle_returns_400(monkeypatch):
    app.state.db = FakeDatabase()
    app.state.rca_provider = FakeProvider()
    app.state.settings = get_settings()
    _patch_run_turn(monkeypatch)
    client = TestClient(app)

    response = client.post("/api/rca/sessions", json={"cycle_id": 999, "query_id": 1})

    assert response.status_code == 400


def test_list_sessions_endpoint(monkeypatch):
    db = FakeDatabase()
    app.state.db = db
    app.state.rca_provider = FakeProvider()
    app.state.settings = get_settings()
    _patch_run_turn(monkeypatch)
    client = TestClient(app)
    client.post("/api/rca/sessions", json={"cycle_id": 62, "query_id": 1})

    response = client.get("/api/rca/sessions")

    assert response.status_code == 200
    assert response.json() == [
        {"id": 1, "cycle_id": 62, "query_id": 1, "total_input_tokens": 100, "total_output_tokens": 50}
    ]


def test_get_messages_endpoint(monkeypatch):
    db = FakeDatabase()
    app.state.db = db
    app.state.rca_provider = FakeProvider()
    app.state.settings = get_settings()
    _patch_run_turn(monkeypatch)
    client = TestClient(app)
    client.post("/api/rca/sessions", json={"cycle_id": 62, "query_id": 1})

    response = client.get("/api/rca/sessions/1/messages")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_send_followup_message(monkeypatch):
    db = FakeDatabase()
    app.state.db = db
    app.state.rca_provider = FakeProvider()
    app.state.settings = get_settings()
    _patch_run_turn(monkeypatch)
    client = TestClient(app)
    client.post("/api/rca/sessions", json={"cycle_id": 62, "query_id": 1})
    _patch_run_turn(monkeypatch, text="Follow-up answer.")

    response = client.post("/api/rca/sessions/1/messages", json={"message": "why though?"})

    assert response.status_code == 200
    assert response.json()["content"] == "Follow-up answer."
    assert len(db.messages[1]) == 4


def test_send_followup_unknown_session_returns_404(monkeypatch):
    app.state.db = FakeDatabase()
    app.state.rca_provider = FakeProvider()
    app.state.settings = get_settings()
    _patch_run_turn(monkeypatch)
    client = TestClient(app)

    response = client.post("/api/rca/sessions/999/messages", json={"message": "hi"})

    assert response.status_code == 404


def test_delete_session_endpoint(monkeypatch):
    db = FakeDatabase()
    app.state.db = db
    app.state.rca_provider = FakeProvider()
    app.state.settings = get_settings()
    _patch_run_turn(monkeypatch)
    client = TestClient(app)
    client.post("/api/rca/sessions", json={"cycle_id": 62, "query_id": 1})

    response = client.delete("/api/rca/sessions/1")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}
    assert 1 not in db.sessions


def test_delete_session_unknown_returns_404():
    app.state.db = FakeDatabase()
    client = TestClient(app)

    response = client.delete("/api/rca/sessions/999")

    assert response.status_code == 404


def test_list_eval_queries_endpoint():
    app.state.db = FakeDatabase()
    client = TestClient(app)

    response = client.get("/api/eval/queries")

    assert response.status_code == 200
    assert response.json() == [{"id": 1, "query_text": "serial number?", "query_type": "extractive"}]
