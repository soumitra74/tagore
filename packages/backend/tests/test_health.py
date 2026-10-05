from fastapi.testclient import TestClient

from app.main import app


class FakeDatabase:
    async def list_tables(self) -> list[str]:
        return ["documents", "eval_runs"]


class FailingDatabase:
    async def list_tables(self) -> list[str]:
        raise RuntimeError("connection lost")


def test_health_returns_ok_with_table_list():
    # No `with` block: this avoids triggering FastAPI's startup lifespan,
    # which wires app.state.db to a real Postgres connection. Setting
    # app.state.db directly keeps this test stable without one.
    app.state.db = FakeDatabase()
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "tables": ["documents", "eval_runs"]}


def test_health_reports_error_when_db_unavailable():
    app.state.db = FailingDatabase()
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["tables"] == []
    assert "connection lost" in body["error"]
